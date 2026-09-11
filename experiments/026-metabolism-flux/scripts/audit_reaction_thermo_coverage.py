# experiments/026-metabolism-flux/scripts/audit_reaction_thermo_coverage.py
# [[experiments.026-metabolism-flux.scripts.audit_reaction_thermo_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/audit_reaction_thermo_coverage.py

r"""Reaction-level thermodynamic coverage: three routes, what each misses, and why.

The coverage ladder elsewhere compares two tables at the reaction level and stops there.
Three questions it does not answer are asked here, because reaction coverage is what the
second-law hinge consumes and a metabolite-level number overstates it.

**1. There are THREE routes to a reaction energy, not two, and they disagree.** yeast-GEM
ships a reaction table AND a metabolite table, and the reaction table is not the sum of the
metabolite one: they were built from different sources with different matching keys (see
the provenance record below). Our recomputation is a third route. All three are counted
here, and the cross-tabulation says which reactions only one of them can price.

**2. Why a reaction cannot be recovered.** A reaction needs EVERY participant priced, so one
unpriced metabolite kills it however common the rest are. Ranking metabolites by how many
otherwise-priceable reactions each one blocks turns a diffuse coverage gap into a short
list: if a handful of ubiquitous species block hundreds of reactions, resolving those few
structures is worth more than any mapping work.

**3. Where the shipped values come from.** Read out of yeast-GEM's own curation script
``code/modelCuration/v8_7_0.m`` (PR #330) rather than inferred. Metabolites: yETFL matched
**by metabolite name**, then ModelSEED matched by KEGG compound id. Reactions: yETFL matched
by reaction id, then dG-predictor assigned **by row position** for a fixed index block. That
is three methods behind one column, which is the reason a comparison against a single
consistent method is not expected to agree.

Writes ``results/reaction_thermo_coverage.json`` and
``results/reaction_blocking_metabolites.csv``. Runs in the torchcell env (cobra only; no
eQuilibrator call is made, everything is read from the built tables).
"""

import json
import os
import os.path as osp
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")

#: yeast-GEM writes this where it has no value, rather than leaving the cell empty. Reading
#: it as a number is how a missing energy becomes a 10,000,000 kJ/mol reaction.
SHIPPED_SENTINEL = 1e7

#: Provenance of the shipped column, quoted from yeast-GEM's own curation script rather
#: than inferred from the values. Every claim here is a line of that file.
SHIPPED_PROVENANCE = {
    "source_file": "code/modelCuration/v8_7_0.m",
    "pull_request": "yeast-GEM PR #330",
    "metabolites": {
        "primary": "yETFL (yeast8_thermo_curated.mat, Zenodo record 4778047)",
        "primary_match_key": "metNames, the metabolite NAME string",
        "fallback": "ModelSEED",
        "fallback_match_key": "kegg.compound identifier",
        "note_in_source": "'Gather ModelSEED data via get_seed_data.py, CSV obtained how?'",
    },
    "reactions": {
        "primary": "yETFL (rxnDeltaGR)",
        "primary_match_key": "reaction id",
        "fallback": "dG-predictor (group contribution)",
        "fallback_match_key": (
            "ROW POSITION: model.rxnDeltaG(3983:4062) = dgpred_rxnG.detaG(3983:4062), "
            "which assumes the two tables are row-aligned"
        ),
        "note_in_source": (
            "'no deltaG is found for model.rxns(3983:4131). Current dG-predictor dataset "
            "has data for model.rxns(1:4062).'"
        ),
    },
    "consequence": (
        "The reaction column and the metabolite column come from different sources under "
        "different keys, so the reaction table is not the sum of the metabolite table and "
        "neither is a single self-consistent thermodynamic method."
    ),
}


def shipped_series(name: str) -> pd.Series:
    """One shipped table as id -> value, with the sentinel turned into NaN."""
    frame = pd.read_csv(osp.join(GEM_DIR, "data", "databases", name))
    values = pd.to_numeric(frame["Var2"], errors="coerce")
    values[values >= SHIPPED_SENTINEL] = np.nan
    return pd.Series(values.to_numpy(), index=frame["Var1"].astype(str))


def main() -> None:
    import cobra

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    reactions = [r.id for r in model.reactions]
    participants = {r.id: {m.id for m in r.metabolites} for r in model.reactions}

    shipped_met = shipped_series("model_metDeltaG.csv")
    shipped_rxn = shipped_series("model_rxnDeltaG.csv")
    recomputed = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"))
    recomputed_priced = set(
        recomputed.loc[recomputed["dgf_prime_kj_per_mol"].notna(), "met_id"]
    )
    shipped_met_priced = set(shipped_met.index[shipped_met.notna()])

    def priceable(priced: set[str]) -> set[str]:
        """Reactions all of whose participants carry a value."""
        return {r for r in reactions if participants[r] and participants[r] <= priced}

    by_shipped_rxn = set(shipped_rxn.index[shipped_rxn.notna()]) & set(reactions)
    by_shipped_met = priceable(shipped_met_priced)
    by_recomputed = priceable(recomputed_priced)

    routes = {
        "shipped_reaction_column": by_shipped_rxn,
        "shipped_metabolite_sum": by_shipped_met,
        "equilibrator_recomputation": by_recomputed,
    }
    n_reactions = len(reactions)
    report: dict = {
        "n_reactions": n_reactions,
        "n_metabolites": len(model.metabolites),
        "shipped_provenance": SHIPPED_PROVENANCE,
        "routes": {
            name: {"n": len(covered), "frac": len(covered) / n_reactions}
            for name, covered in routes.items()
        },
    }

    # The cross-tabulation. A reaction priced by exactly one route is the interesting cell:
    # it is either a gain or a gap depending on which route that is.
    membership = Counter()
    for r in reactions:
        key = tuple(sorted(name for name, covered in routes.items() if r in covered))
        membership[key or ("none",)] += 1
    report["route_overlap"] = {
        "+".join(key): count for key, count in sorted(membership.items(), key=lambda kv: -kv[1])
    }
    report["only_recomputation"] = len(by_recomputed - by_shipped_met)
    report["only_shipped_metabolites"] = len(by_shipped_met - by_recomputed)
    # Two different unions, and they differ by design rather than by error. Taking the
    # union of the two REACTION sets asks which reactions either route can price on its
    # own. Merging the two METABOLITE tables first and then pricing asks what a combined
    # table can do, and it is larger: a reaction can draw one participant from each table
    # and so be priceable under the merge while neither route prices it alone.
    by_merged_metabolites = priceable(shipped_met_priced | recomputed_priced)
    report["union_of_reaction_sets"] = len(by_shipped_met | by_recomputed)
    report["priceable_under_merged_metabolite_table"] = len(by_merged_metabolites)
    report["reactions_needing_both_tables"] = len(
        by_merged_metabolites - (by_shipped_met | by_recomputed)
    )

    # The shipped reaction column against the sum of the shipped metabolite column, which
    # is the internal-consistency question at the level of WHICH reactions, not values.
    report["shipped_reaction_column_not_reproducible_from_its_own_metabolites"] = len(
        by_shipped_rxn - by_shipped_met
    )

    # WHY a reaction cannot be recovered: rank the metabolites that block the most
    # reactions the shipped metabolite table can already price.
    recoverable_target = by_shipped_met - by_recomputed
    blocker_counts: Counter = Counter()
    for r in recoverable_target:
        for met in participants[r] - recomputed_priced:
            blocker_counts[met] += 1
    names = {m.id: m.name for m in model.metabolites}
    formulas = {m.id: (m.formula or "") for m in model.metabolites}
    blockers = pd.DataFrame(
        [
            {
                "met_id": met,
                "name": names.get(met, ""),
                "formula": formulas.get(met, ""),
                "n_reactions_blocked": count,
            }
            for met, count in blocker_counts.most_common()
        ]
    )
    blockers.to_csv(osp.join(RESULTS, "reaction_blocking_metabolites.csv"), index=False)

    # How concentrated is the gap? If a few metabolites carry it, the fix is short.
    cumulative = blockers["n_reactions_blocked"].cumsum() if len(blockers) else pd.Series()
    # A reaction can be blocked by several metabolites at once, so a cumulative sum over
    # blocked-counts double counts. The honest measure resolves the top-k as a SET.
    def reactions_freed_by(top_k: int) -> int:
        freed = set(blockers["met_id"].head(top_k)) | recomputed_priced
        return len({r for r in recoverable_target if participants[r] <= freed})

    report["recovery_gap"] = {
        "n_reactions_shipped_can_price_and_we_cannot": len(recoverable_target),
        "n_distinct_blocking_metabolites": int(len(blockers)),
        "n_blocking_metabolites_with_placeholder_formula": int(
            sum(
                1
                for f in blockers["formula"]
                if any(token in f for token in ("R", "X", "*"))
            )
        )
        if len(blockers)
        else 0,
        "reactions_freed_by_top_10_metabolites": reactions_freed_by(10),
        "reactions_freed_by_top_25_metabolites": reactions_freed_by(25),
        "reactions_freed_by_top_50_metabolites": reactions_freed_by(50),
        "top_blockers": blockers.head(12).to_dict("records") if len(blockers) else [],
    }

    # And the reverse gain: what the recomputation adds that the shipped metabolites cannot.
    gained = by_recomputed - by_shipped_met
    gained_blockers: Counter = Counter()
    for r in gained:
        for met in participants[r] - shipped_met_priced:
            gained_blockers[met] += 1
    report["gain"] = {
        "n_reactions_only_recomputation_can_price": len(gained),
        "top_metabolites_the_shipped_table_lacks": [
            {"met_id": met, "name": names.get(met, ""), "n_reactions": count}
            for met, count in gained_blockers.most_common(10)
        ],
    }

    with open(osp.join(RESULTS, "reaction_thermo_coverage.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"{n_reactions} reactions")
    for name, block in report["routes"].items():
        print(f"  {name:32s} {block['n']:5d}  {block['frac']:6.2%}")
    print("\noverlap:")
    for key, count in report["route_overlap"].items():
        print(f"  {count:5d}  {key}")
    print(
        f"\nshipped-metabolite route can price but we cannot: "
        f"{report['recovery_gap']['n_reactions_shipped_can_price_and_we_cannot']}, "
        f"blocked by {report['recovery_gap']['n_distinct_blocking_metabolites']} metabolites"
    )
    for row in report["recovery_gap"]["top_blockers"]:
        print(
            f"  {row['n_reactions_blocked']:5d}  {row['met_id']:8s} {row['name'][:44]:44s}"
            f" {row['formula']}"
        )
    print(
        f"  top 10 would free {report['recovery_gap']['reactions_freed_by_top_10_metabolites']}, "
        f"top 25 {report['recovery_gap']['reactions_freed_by_top_25_metabolites']}, "
        f"top 50 {report['recovery_gap']['reactions_freed_by_top_50_metabolites']}"
    )
    print(f"\nwrote {osp.join(RESULTS, 'reaction_thermo_coverage.json')}")


if __name__ == "__main__":
    main()
