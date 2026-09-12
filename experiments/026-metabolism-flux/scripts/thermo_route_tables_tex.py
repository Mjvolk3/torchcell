# experiments/026-metabolism-flux/scripts/thermo_route_tables_tex.py
# [[experiments.026-metabolism-flux.scripts.thermo_route_tables_tex]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_route_tables_tex.py

r"""Emit the LaTeX route tables of the 026 kinetics-thermodynamics document.

Five tables, all written into ``notes-tex/026-kinetics-thermodynamics/tables/``:

``thermo_metabolite_routes.tex``
    One row per route that produced a formation energy: what it does, what it needs as
    input, its evidence class, how many of the 2,806 metabolites it priced, and the
    script or note that documents it.
``thermo_reaction_routes.tex``
    The same for the routes that produced a reaction energy.
``thermo_unpriced_classes.tex``
    One row per failure class, with the metabolites and the reactions it holds and
    what it would take to move.
``thermo_curation_records.tex``
    The 16 hand-curated records, their admission and what each closed.
``thermo_accession_mispricings.tex``
    The eight species the composition check found priced as a different molecule.

The COUNTS come from ``results/reaction_thermo_coverage_final.json``,
``results/thermo_exclusions.json`` and ``results/thermo_curation_closure.json``. The
PROSE columns are declared here, because a route's description is documentation rather
than a measurement; regenerating the tables is what keeps the two in step, and a route
that appears in a result file with no entry here raises rather than being dropped.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_route_tables_tex.py
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

COVERAGE_JSON = osp.join(RESULTS, "reaction_thermo_coverage_final.json")
PARAMETER_COVERAGE_JSON = osp.join(RESULTS, "parameter_coverage.json")
EXCLUSIONS_JSON = osp.join(RESULTS, "thermo_exclusions.json")
CLOSURE_JSON = osp.join(RESULTS, "thermo_curation_closure.json")


class RouteDoc(BaseModel):
    """The prose half of one route's row."""

    model_config = ConfigDict(extra="forbid")

    #: Name as it reads in the document, and the term the controlled vocabulary defines.
    display: str
    #: What the route does to turn an input into a formation or reaction energy.
    does: str
    #: What it needs to be given.
    needs: str
    #: measured, tabulated, estimated or curated.
    evidence: str
    #: The committed script or dendron note that documents it.
    documented_by: str


class ReasonDoc(BaseModel):
    """The prose half of one failure class."""

    model_config = ConfigDict(extra="forbid")

    display: str
    what_it_means: str
    remedy: str


#: Routes that can price a METABOLITE. Order is the pipeline's stage order, not the count
#: order, so the table reads as the sequence a metabolite is put through.
METABOLITE_ROUTES: dict[str, RouteDoc] = {
    "cache_accession": RouteDoc(
        display="cache, by accession",
        does=(
            "reads an eQuilibrator compound the model's own identifier resolves, takes its"
            " component-contribution formation energy and transforms it to the"
            " metabolite's compartment"
        ),
        needs="a MetaNetX, ChEBI, KEGG or BiGG identifier on the metabolite",
        evidence="estimated",
        documented_by="build\\_equilibrator\\_thermo.py, thermo\\_accession\\_sweep.py",
    ),
    "cache_structure": RouteDoc(
        display="cache, by written structure",
        does=(
            "writes a structure from the metabolite's name, hashes it to an InChIKey and"
            " finds a compound eQuilibrator already holds under that key"
        ),
        needs="a name stating the acyl composition, and a headgroup template",
        evidence="estimated",
        documented_by="lipid\\_structure\\_writer.py",
    ),
    "external_pka": RouteDoc(
        display="external pKa cascade",
        does=(
            "creates a compound eQuilibrator has never seen, from the pH 7 major"
            " microspecies plus a predicted pKa ladder, then decomposes and prices it"
        ),
        needs="a structure and a pKa ladder from the selected cascade",
        evidence="estimated",
        documented_by="external\\_pka\\_apply\\_selection.py",
    ),
    "literature_table": RouteDoc(
        display="CODATA ion table",
        does=(
            "reads a tabulated single-ion formation energy, shifts it to eQuilibrator's"
            " gauge and applies the Debye-Huckel term"
        ),
        needs="a monatomic ion whose species CODATA evaluates",
        evidence="tabulated",
        documented_by="fetch\\_ion\\_formation\\_energies.py, thermo\\_ion\\_gauge\\_check.py",
    ),
    "composition_surrogate": RouteDoc(
        display="composition surrogate",
        does=(
            "transfers the energy of a cache compound sharing the heavy-atom composition,"
            " admitted only where every such compound agrees to 5 kJ/mol"
        ),
        needs="a definite formula the cache can decompose elsewhere",
        evidence="estimated",
        documented_by="improve\\_parameter\\_coverage.py",
    ),
    "repeat_unit_surrogate": RouteDoc(
        display="repeat-unit surrogate",
        does=(
            "prices one polymer residue as the monomer minus water minus half a"
            " calibrated hydrolysis energy"
        ),
        needs="a hydrolysis reaction in the model that names the monomer",
        evidence="estimated",
        documented_by="thermo\\_polymer\\_repeat\\_unit.py",
    ),
    "curated_structure": RouteDoc(
        display="curated structure or convention",
        does=(
            "prices a species from a structure or a polymer convention a curation record"
            " fixes, with sources and a sha256 per source"
        ),
        needs="an admitted curation record of kind structure or polymer convention",
        evidence="curated",
        documented_by="thermo\\_curation\\_price.py, results/curation/",
    ),
    "carrier_potential": RouteDoc(
        display="carrier potential",
        does=(
            "gauges an electron carrier pair, oxidized form at zero and reduced form at"
            " $-nFE'^{\\circ}$, so the arbitrary gauge cancels in any reaction that"
            " conserves the carrier"
        ),
        needs="a measured midpoint potential for the couple",
        evidence="measured",
        documented_by="thermo\\_curation\\_price.py, results/curation/cytochrome\\_c.json",
    ),
    "shipped_table": RouteDoc(
        display="shipped metabolite column",
        does=(
            "reads yeast-GEM's own \\texttt{model\\_metDeltaG.csv}, converted out of"
            " kilocalories at parse time"
        ),
        needs="a row in that file for this metabolite",
        evidence="tabulated",
        documented_by="yeast9\\_delta\\_g\\_provenance.py",
    ),
}

#: Routes that can price a REACTION. The metabolite routes reappear here for a reaction
#: all of whose participants came from one of them.
REACTION_ONLY_ROUTES: dict[str, RouteDoc] = {
    "mixed": RouteDoc(
        display="mixed routes",
        does=(
            "sums participants priced by more than one route, so the reaction's"
            " uncertainty belongs to no single route"
        ),
        needs="every participant priced by something",
        evidence="estimated",
        documented_by="thermo\\_pipeline\\_yeast9.py",
    ),
    "cancelled_participants": RouteDoc(
        display="cancellation",
        does=(
            "drops a species appearing unchanged on both sides, so a reaction closes"
            " without that species ever being priced"
        ),
        needs="an unpriced species identical in name, formula and charge on both sides",
        evidence="estimated",
        documented_by="thermo\\_pipeline\\_yeast9.py",
    ),
    "conserved_moiety": RouteDoc(
        display="conserved moiety",
        does=(
            "decomposes every participant leniently and prices the group difference where"
            " the atoms no group covers match as a multiset across the two sides"
        ),
        needs="a curated structure for each participant and a moiety that cancels",
        evidence="estimated",
        documented_by="thermo\\_curation\\_price.py",
    ),
    "shipped_reaction_column": RouteDoc(
        display="shipped reaction column",
        does="reads yeast-GEM's \\texttt{model\\_rxnDeltaG.csv} directly",
        needs="a row in that file, with the 33 dGPredictor rows exempt from the conversion",
        evidence="tabulated",
        documented_by="thermo\\_dgpredictor\\_rows.py",
    ),
}

REASONS: dict[str, ReasonDoc] = {
    "placeholder_formula": ReasonDoc(
        display="placeholder formula",
        what_it_means=(
            "an R, X or $*$ in the formula: the species is a family of molecules and no"
            " single formation energy exists for it"
        ),
        remedy="none; the ceiling",
    ),
    "pseudo_metabolite": ReasonDoc(
        display="pseudo-metabolite",
        what_it_means=(
            "a modeling device, a pool, a SLIME backbone or a difference formula, rather"
            " than a molecule"
        ),
        remedy="none; the ceiling",
    ),
    "placeholder_structure": ReasonDoc(
        display="placeholder structure",
        what_it_means=(
            "the only structure that resolves carries an attachment point, so the species"
            " is a residue inside a macromolecule"
        ),
        remedy="none; the ceiling",
    ),
    "group_decomposition_failed": ReasonDoc(
        display="decomposition refused",
        what_it_means=(
            "the structure exists and component contribution refuses it, so a pKa"
            " predictor changes nothing"
        ),
        remedy="a decomposition, or a tabulated value",
    ),
    "group_in_null_space": ReasonDoc(
        display="group in the null space",
        what_it_means=(
            "the structure decomposes and one group is in the component-contribution null"
            " space, so the estimate has infinite variance"
        ),
        remedy="a measurement of a compound carrying that group",
    ),
    "no_structure_available": ReasonDoc(
        display="no structure",
        what_it_means=(
            "a definite formula with no structure in any namespace and no writer that"
            " composes one from the name"
        ),
        remedy="a structure from a namespace or a curator",
    ),
    "recorded_not_admitted": ReasonDoc(
        display="recorded, not admitted",
        what_it_means=(
            "a curation record exists and its admission is computed or provisional, so it"
            " is reported and not priced"
        ),
        remedy="a measured or tabulated source",
    ),
    "mixed_reasons": ReasonDoc(
        display="more than one reason",
        what_it_means="the reaction is blocked by species in two or more of the classes above",
        remedy="every blocking class resolved",
    ),
    "non_chemical": ReasonDoc(
        display="non-chemical reaction",
        what_it_means=(
            "an exchange, demand, sink or biomass pseudoreaction, not elementally balanced,"
            " so a standard reaction energy has no meaning for it"
        ),
        remedy="none; excluded from the chemical denominator",
    ),
}

#: Short forms so the curation table fits the 182 mm text block.
KIND_SHORT = {
    "carrier_potential": "carrier",
    "polymer_convention": "polymer",
    "structure": "structure",
    "pka_ladder": "pKa ladder",
    "formula_correction": "formula",
    "not_found": "not found",
}
DECOMPOSITION_SHORT = {
    "decomposes": "yes",
    "refused": "refused",
    "mixed": "mixed",
    "not attempted": "n/a",
}

HEADER = (
    "%% notes-tex/026-kinetics-thermodynamics/tables/{name}\n"
    "%% AUTO-GENERATED -- do not hand-edit. Regenerate with\n"
    "%%   PYTHONPATH=. python experiments/026-metabolism-flux/scripts/"
    "thermo_route_tables_tex.py\n"
    "%% SOURCE: {source}\n"
)


def _num(value: int) -> str:
    """Thousands separator LaTeX can typeset without a space after the comma."""
    return f"{value:,}".replace(",", "{,}")


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def _write(name: str, source: str, body: str) -> None:
    path = osp.join(TABLE_DIR, name)
    with open(path, "w") as handle:
        handle.write(HEADER.format(name=name, source=source) + body)
    print(f"wrote {path}")


def _route_rows(
    docs: dict[str, RouteDoc], counts: dict[str, int], kind: str
) -> list[tuple[RouteDoc, int]]:
    rows: list[tuple[RouteDoc, int]] = []
    for route, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        if route == "none":
            continue
        if route not in docs:
            raise KeyError(
                f"{kind} route {route!r} has no RouteDoc in thermo_route_tables_tex.py; "
                "a route may not reach the document undocumented"
            )
        rows.append((docs[route], count))
    return rows


def _route_table(
    rows: list[tuple[RouteDoc, int]], caption: str, label: str, unit: str, short: str
) -> str:
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        f"  \\caption[{short}]{{{caption}}}",
        f"  \\label{{tab:{label}}}",
        "  \\begin{tabular}{@{}p{0.135\\textwidth}p{0.30\\textwidth}"
        "p{0.19\\textwidth}lr@{}}",
        "    \\toprule",
        f"    Route & What it does & What it needs & Evidence & {unit} \\\\",
        "    \\midrule",
    ]
    for doc, count in rows:
        lines.append(
            f"    {doc.display} & {doc.does} & {doc.needs} & {doc.evidence} & "
            f"{_num(count)} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _documented_by_table(rows: list[tuple[RouteDoc, int]]) -> str:
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Route provenance]{Which committed script documents and produces each"
        " route. A route present in \\file{results/thermo\\_pipeline\\_yeast9.json} with no"
        " entry here makes \\file{thermo\\_route\\_tables\\_tex.py} raise, so the table"
        " cannot fall behind the pipeline.}",
        "  \\label{tab:route-provenance}",
        "  \\begin{tabular}{@{}p{0.24\\textwidth}p{0.66\\textwidth}@{}}",
        "    \\toprule",
        "    Route & Produced and documented by \\\\",
        "    \\midrule",
    ]
    seen: set[str] = set()
    for doc, _count in rows:
        if doc.display in seen:
            continue
        seen.add(doc.display)
        lines.append(f"    {doc.display} & \\texttt{{{doc.documented_by}}} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


#: Terms the recovery sections use that are not a route or a failure class. The vocabulary
#: table is the union of these with both of those, sorted alphabetically, so a term cannot
#: reach the prose without a definition.
EXTRA_TERMS: dict[str, str] = {
    "admission": (
        "the strongest claim a curation record's sources support: measured, tabulated,"
        " computed or provisional. Only measured and tabulated reach a served table"
    ),
    "catalytic unit": (
        "one reaction and the gene product complex that catalyzes it, the key a turnover"
        " number and a Michaelis constant are stored under"
    ),
    "component contribution": (
        "eQuilibrator's estimator: decompose a structure into groups, fit group energies"
        " against a measured reaction corpus, and return a formation energy with the"
        " covariance that fit implies"
    ),
    "evidence class": (
        "what a value rests on, one of measured, tabulated, estimated by component"
        " contribution, or curated"
    ),
    "flux-carrying": (
        "a reaction whose flux-variability width is nonzero under the wild-type medium,"
        " so the layer can express a flux through it"
    ),
    "microspecies": (
        "one protonation state of a compound. The reference microspecies is the state the"
        " caller draws, and choosing the neutral parent rather than the pH 7 major"
        " microspecies moves a formation energy by hundreds of kJ/mol"
    ),
    "non-chemical reaction": (
        "an exchange, demand, sink or biomass pseudoreaction, excluded from the chemical"
        " denominator because it is not elementally balanced"
    ),
    "pKa ladder": (
        "the list of macroscopic dissociation constants inside pH 0 to 14 that fixes how"
        " a compound's protonation states are populated, and therefore its Legendre"
        " transform"
    ),
    "permanently closed": (
        "a reaction no pKa predictor, structure writer, decomposer or literature table"
        " can price, because every species blocking it is not a molecule"
    ),
    "priced": (
        "carrying a transformed formation energy at the species' own compartment"
        " conditions; a reaction is priced when every participant is, or when a"
        " reaction-level route fills it directly"
    ),
    "target": (
        "one distinct structure submitted to the external pKa route. Several yeast-GEM"
        " species can share one target"
    ),
}


def _vocabulary_table() -> str:
    """Every term the recovery sections use, defined before use and alphabetical."""
    entries: dict[str, str] = dict(EXTRA_TERMS)
    for doc in {**METABOLITE_ROUTES, **REACTION_ONLY_ROUTES}.values():
        entries[doc.display] = f"a route. {doc.does[0].upper()}{doc.does[1:]}"
    for reason in REASONS.values():
        entries[reason.display] = f"a failure class. {reason.what_it_means}"
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Controlled vocabulary]{Every term the recovery sections use, in"
        " alphabetical order. A route is a way of producing a value, a failure class is a"
        " reason there is none, and the two partition every species and every reaction in"
        " the model.}",
        "  \\label{tab:thermo-vocabulary}",
        "  \\begin{tabular}{@{}p{0.24\\textwidth}p{0.70\\textwidth}@{}}",
        "    \\toprule",
        "    Term & Definition \\\\",
        "    \\midrule",
    ]
    for term in sorted(entries, key=str.lower):
        lines.append(f"    {term} & {entries[term]} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


#: What each ladder step supplies beyond a number, which is where the sources differ most.
LADDER_STEP = {
    "shipped_only": (
        "yeast-GEM shipped column",
        "kcal-scale point values, no uncertainty, this model only",
    ),
    "recomputed_only": (
        "eQuilibrator recomputation, baseline",
        "covariance of rank 590, any organism, any compound with a structure",
    ),
    "union_shipped_recomputed": (
        "union of the two",
        "covariance where the recomputation reaches, point values elsewhere",
    ),
    "union_plus_tight_surrogate": (
        "union plus the tight surrogate",
        "as the union, plus 39 species a same-composition surrogate determines",
    ),
    "union_plus_all_surrogate_NOT_RECOMMENDED": (
        "union plus every surrogate",
        "refused: the median surrogate spread is 52.6 kJ/mol, which orients nothing",
    ),
}


def _ladder_table(
    parameter_coverage: dict[str, object], coverage: dict[str, object]
) -> str:
    """The coverage ladder, ending on the pipeline as built."""
    gem = cast(dict[str, int], parameter_coverage["gem"])
    n_reactions = int(gem["n_reactions"])
    n_metabolites = int(gem["n_metabolites"])
    flux = cast(dict[str, float], parameter_coverage["flux_weighted"])
    n_flux = int(flux["n_reactions_with_nonzero_fva_width"])
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Coverage ladder]{Every source of a formation energy counted against"
        f" the whole model, {_num(n_reactions)} reactions and {_num(n_metabolites)}"
        " metabolites. A reaction counts only when every participant carries a value,"
        " which is the level the flux layer consumes, since the second-law hinge needs the"
        " whole stoichiometric sum; the metabolite column sits beside it because a"
        " metabolite-level figure is the one usually quoted and it overstates what is"
        " usable. The last row is the pipeline as built, which reaches every source above"
        " and eight recovery routes besides"
        " (Table~\\ref{tab:thermo-metabolite-routes}). The shipped row's values are on the"
        " wrong scale, being yETFL's in kcal\\,mol$^{-1}$ and labeled kilojoules"
        " (Section~\\ref{sec:shipped-provenance}), so its count is a count of numbers"
        " rather than of reactions a consumer can read without a conversion.}",
        "  \\label{tab:ladder}",
        "  \\begin{tabular}{@{}lrrrrp{0.32\\textwidth}@{}}",
        "    \\toprule",
        "    Source & Rxns & \\% & Mets & \\% & Supplies \\\\",
        "    \\midrule",
    ]
    for step in cast(list[dict[str, object]], parameter_coverage["delta_g_ladder"]):
        name = cast(str, step["step"])
        display, supplies = LADDER_STEP[name]
        n_rxn = int(cast(int, step["n_reactions_priced"]))
        n_met = int(cast(int, step["n_metabolites_priced"]))
        lines.append(
            f"    {display} & {_num(n_rxn)} & {100.0 * n_rxn / n_reactions:.2f} & "
            f"{_num(n_met)} & {100.0 * n_met / n_metabolites:.2f} & {supplies} \\\\"
        )
    n_rxn = int(cast(int, coverage["n_reactions_priced"]))
    n_met = int(cast(int, coverage["n_metabolites_priced"]))
    n_flux_priced = int(cast(dict[str, float], coverage["flux_carrying"])["n_priced"])
    lines += [
        "    \\midrule",
        f"    \\textbf{{as built}} & \\textbf{{{_num(n_rxn)}}} & "
        f"\\textbf{{{100.0 * n_rxn / n_reactions:.2f}}} & \\textbf{{{_num(n_met)}}} & "
        f"\\textbf{{{100.0 * n_met / n_metabolites:.2f}}} & "
        "one covariance-carrying estimate wherever one exists, and a named route and"
        " provenance record on every value \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{0.3em}\\\\",
        f"  {{\\footnotesize Over the {_num(n_flux)} reactions with nonzero"
        " flux-variability width under the wild-type medium, the last row is"
        f" {_num(n_flux_priced)}, {100.0 * n_flux_priced / n_flux:.2f} percent.}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    coverage = _read(COVERAGE_JSON)
    exclusions = _read(EXCLUSIONS_JSON)
    closure = _read(CLOSURE_JSON)

    met_routes = cast(dict[str, int], coverage["metabolite_routes"])
    rxn_routes = cast(dict[str, int], coverage["reaction_routes"])
    n_met_priced = int(cast(int, coverage["n_metabolites_priced"]))
    n_met_unpriced = int(cast(int, coverage["n_metabolites"])) - n_met_priced
    n_reactions = int(cast(int, coverage["n_reactions"]))
    n_rxn_priced = int(cast(int, coverage["n_reactions_priced"]))

    met_rows = _route_rows(METABOLITE_ROUTES, met_routes, "metabolite")
    reaction_docs = {**METABOLITE_ROUTES, **REACTION_ONLY_ROUTES}
    rxn_rows = _route_rows(reaction_docs, rxn_routes, "reaction")

    _write(
        "thermo_vocabulary.tex",
        "hand-authored definitions declared in thermo_route_tables_tex.py, assembled "
        "from the same RouteDoc and ReasonDoc records the route tables are built from",
        _vocabulary_table(),
    )
    _write(
        "thermo_coverage_ladder.tex",
        "results/parameter_coverage.json, delta_g_ladder and flux_weighted "
        "(improve_parameter_coverage.py); the last row from "
        "results/reaction_thermo_coverage_final.json (thermo_pipeline_yeast9.py)",
        _ladder_table(_read(PARAMETER_COVERAGE_JSON), coverage),
    )

    _write(
        "thermo_metabolite_routes.tex",
        "results/reaction_thermo_coverage_final.json, metabolite_routes; prose column "
        "declared in thermo_route_tables_tex.py",
        _route_table(
            met_rows,
            "Every route that produced a formation energy, and how many of yeast-GEM"
            " 9.0.2's 2{,}806 metabolites it priced. The nine routes sum to the"
            f" {_num(n_met_priced)} priced species; the {_num(n_met_unpriced)} with no"
            " value are Table~\\ref{tab:thermo-unpriced-classes}. Evidence class is the"
            " strongest thing the number rests on, so a surrogate is estimated rather"
            " than tabulated: its value is component contribution on another molecule.",
            "thermo-metabolite-routes",
            "Species",
            "Formation-energy routes",
        ),
    )
    _write(
        "thermo_reaction_routes.tex",
        "results/reaction_thermo_coverage_final.json, reaction_routes; prose column "
        "declared in thermo_route_tables_tex.py",
        _route_table(
            rxn_rows,
            "Every route that produced a reaction energy, over yeast-GEM 9.0.2's"
            f" {_num(n_reactions)} reactions, {_num(n_rxn_priced)} of them priced. A"
            " reaction is priced only when every participant carries a formation energy,"
            " or when a reaction-level route fills it directly, so the largest entry is"
            " the metabolite route its participants all came from and the second is the"
            " reactions assembled from more than one.",
            "thermo-reaction-routes",
            "Reactions",
            "Reaction-energy routes",
        ),
    )
    _write(
        "thermo_route_provenance.tex",
        "prose declared in thermo_route_tables_tex.py; route list from "
        "results/reaction_thermo_coverage_final.json",
        _documented_by_table(met_rows + rxn_rows),
    )

    met_reasons = cast(dict[str, int], coverage["metabolite_reasons"])
    rxn_by_reason = cast(dict[str, int], exclusions["reactions_blocked_by_reason"])
    rxn_sole = cast(dict[str, int], coverage["reactions_by_sole_blocking_reason"])
    permanent = set(
        cast(
            dict[str, int],
            cast(dict[str, object], coverage["ceiling_ladder"])[
                "permanently_closed_reasons"
            ],
        )
    )
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Failure classes]{Every class that holds a value back, the species it"
        " holds and the reactions it blocks. \\emph{Species} counts metabolites carrying"
        " that reason; \\emph{blocks} counts reactions with at least one such participant,"
        " so the column double counts a reaction blocked by two classes; \\emph{sole}"
        " counts reactions this class alone blocks and partitions the 330 unpriced"
        " reactions exactly. A class marked permanent is one no pKa predictor, structure"
        " writer, decomposer or literature table can move, so the sum of the permanent"
        " rows is the ceiling for any method rather than for this one.}",
        "  \\label{tab:thermo-unpriced-classes}",
        "  \\begin{tabular}{@{}p{0.16\\textwidth}p{0.34\\textwidth}"
        "p{0.17\\textwidth}rrr@{}}",
        "    \\toprule",
        "    Class & What it means & What it would take & Species & Blocks & Sole \\\\",
        "    \\midrule",
    ]
    for reason in sorted(
        set(met_reasons) | set(rxn_sole) | {"non_chemical"},
        key=lambda r: -(met_reasons.get(r, 0) + rxn_sole.get(r, 0)),
    ):
        doc = REASONS[reason]
        mark = (
            "$^{\\dagger}$" if reason in permanent or reason == "non_chemical" else ""
        )
        lines.append(
            f"    {doc.display}{mark} & {doc.what_it_means} & {doc.remedy} & "
            f"{met_reasons.get(reason, 0)} & {rxn_by_reason.get(reason, 0)} & "
            f"{rxn_sole.get(reason, 0)} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{0.3em}\\\\",
        "  {\\footnotesize $^{\\dagger}$ permanent under the admission rule.}",
        "\\end{table}",
        "",
    ]
    _write(
        "thermo_unpriced_classes.tex",
        "results/reaction_thermo_coverage_final.json (metabolite_reasons, "
        "reactions_by_sole_blocking_reason, ceiling_ladder.permanently_closed_reasons) "
        "and results/thermo_exclusions.json (reactions_blocked_by_reason)",
        "\n".join(lines),
    )

    rows = cast(list[dict[str, object]], closure["rows"])
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Curation records]{The 16 hand-curated records. \\emph{Admission} is"
        " the strongest claim the sources support, and only \\emph{measured} and"
        " \\emph{tabulated} reach the table a model reads: a provisional or absent record"
        " is loaded, validated and reported without being priced, which is why five of"
        " them price nothing. \\emph{Closed} counts reactions the record itself closed, with"
        " in parentheses the reactions closed because the species cancels across the"
        " reaction instead. Each record carries its sources with a sha256 per source in"
        " \\file{experiments/026-metabolism-flux/results/curation/}.}",
        "  \\label{tab:thermo-curation}",
        "  \\begin{tabular}{@{}p{0.15\\textwidth}lllrrlp{0.24\\textwidth}@{}}",
        "    \\toprule",
        "    Record & Kind & Admission & Decomp. & Spec. & Src & Closed & Value \\\\",
        "    \\midrule",
    ]
    for row in sorted(rows, key=lambda r: (not r["admitted"], cast(str, r["slug"]))):
        slug = cast(str, row["slug"]).replace("_", "\\_\\-")
        kind = KIND_SHORT[cast(str, row["kind"])]
        decomposition = DECOMPOSITION_SHORT[cast(str, row["decomposition"])]
        value = cast(str, row["value_summary"])
        if value.startswith("SMILES "):
            value = "a curated structure, drawn as one microspecies"
        value = value[:46] + ("..." if len(value) > 46 else "")
        value = value.replace("_", "\\_").replace("'", "$'$").replace(">", "$>$")
        closed = int(cast(int, row["n_reactions_closed"]))
        cancelled = int(cast(int, row["n_reactions_closed_by_cancellation"]))
        closed_text = f"{closed}" + (f" ({cancelled})" if cancelled else "")
        lines.append(
            f"    {slug} & {kind} & {row['admission']} & {decomposition} & "
            f"{len(cast(list[str], row['met_ids']))} & {row['n_sources']} & "
            f"{closed_text} & {value} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    _write(
        "thermo_curation_records.tex",
        "results/thermo_curation_closure.json, rows; records in results/curation/",
        "\n".join(lines),
    )

    sweep = cast(dict[str, object], closure["accession_sweep"])
    others = cast(list[dict[str, object]], sweep["other"])
    priced_by_slug = cast(
        dict[str, int],
        cast(dict[str, object], coverage["curation"])["n_metabolites_priced_by_slug"],
    )
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Accession mispricings]{The eight species the composition check found"
        " priced as a different molecule. The accession route accepted the first"
        " identifier that resolved to a compound carrying a formation energy, whatever"
        " molecule that compound is; checking the resolved structure's composition"
        f" against the model's own formula over all {_num(int(cast(int, sweep['n_checked'])))}"
        f" accession-priced species finds {_num(int(cast(int, sweep['n_match'])))} matches,"
        f" {sweep['n_repeat_unit_multiple']} dolichols carrying exactly one extra isoprene"
        " unit, and these eight. Six are withdrawn; the glycogen pair is re-derived per"
        " residue from a curated polymer convention instead.}",
        "  \\label{tab:thermo-mispricings}",
        "  \\begin{tabular}{@{}llllrl@{}}",
        "    \\toprule",
        "    Species & Name & GEM formula & Priced as & $\\Delta_f G'^{\\circ}$ & Outcome"
        " \\\\",
        "    \\midrule",
    ]
    for row in others:
        met_id = cast(str, row["met_id"])
        slug_priced = (
            met_id in {"s_0773", "s_0774"} and priced_by_slug.get("glycogen", 0) > 0
        )
        outcome = "re-derived" if slug_priced else "withdrawn"
        name = cast(str, row["name"])
        name = name[:26] + ("..." if len(name) > 26 else "")
        lines.append(
            f"    \\texttt{{{met_id.replace('_', chr(92) + '_')}}} & {name} & "
            f"\\texttt{{{cast(str, row['gem_formula'])}}} & "
            f"\\texttt{{{cast(str, row['structure_formula'])}}} & "
            f"{float(cast(float, row['shipped_dgf_prime_kj_per_mol'])):.1f} & {outcome} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    _write(
        "thermo_accession_mispricings.tex",
        "results/thermo_curation_closure.json, accession_sweep.other; full per-species "
        "rows in results/thermo_accession_composition.json",
        "\n".join(lines),
    )


if __name__ == "__main__":
    main()
