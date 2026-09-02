# experiments/026-metabolism-flux/scripts/media_schema_audit.py
# [[experiments.026-metabolism-flux.scripts.media_schema_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/media_schema_audit.py

r"""Prove the data-ontology ``Media`` schema maps onto GEM exchange bounds, or find where it does not.

Run from the repo/worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/media_schema_audit.py

Writes ``experiments/026-metabolism-flux/results/media_schema_audit.json`` and prints it.

What this measures, and why each part is here
---------------------------------------------
1. **Recipe coverage.** Each of the four media our datasets need (SM, SC, SC-URA,
   YPD-approx) is an ontology ``Media`` object; the audit maps it onto yeast-GEM 9.0.2
   exchange bounds and records every component's fate. A component that neither resolves
   nor is excluded by role is a hole in the mapping, so coverage is reported per medium
   rather than summarized.
2. **Which channel did the work.** The mapping tries a compound's own cross-references
   (``chebi_id``) before any name matching, because an identifier is exact. Counting the
   channels says whether the ontology's declared join key is actually carrying the map.
3. **Annotation-channel round trip.** Every resolved metabolite's own ChEBI id is fed
   back as a ``Compound.chebi_id`` and must return the same exchange. Without this the
   annotation path would be untested code, since no ontology component currently carries
   an identifier for it to use.
4. **Pairwise differences.** Which exchanges separate the four media, and by how much.
5. **Growth.** Plain FBA under each medium, so a bound vector that produces a dead cell
   cannot pass as a successful mapping.
6. **What the loaders actually emit.** The four datasets' ``Media`` objects are read back
   out of their loader source (literal + line number verified at run time, so this cannot
   drift silently) and mapped through the same path as the recipes. The gap between a
   loader's object and its recipe is the finding, not a detail.
7. **Sourced vs rescaled supplements.** Suthers' absolute 0.165 mmol/gDW/h against the
   ``glucose_rate * 0.05`` our older scripts compute, at the glucose rate those scripts
   use, so the size of the divergence is a measured number rather than an assertion.
"""

import json
import os
import os.path as osp
import re
from typing import Any

from dotenv import load_dotenv

from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import Media
from torchcell.metabolism.media import (
    FBA_MEDIA,
    ExchangeIndex,
    MediaBounds,
    UptakePolicy,
    build_exchange_index,
    diff_bounds,
    media_to_bounds,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")
OUT_PATH = osp.join(RESULTS_DIR, "media_schema_audit.json")

#: The four datasets whose medium matters here, with the EXACT ``Media(...)`` literal
#: their loader constructs. The literal is verified against the file at run time, so a
#: loader edit surfaces as a hard failure instead of a stale claim in a report.
DATASET_MEDIA_LITERALS: list[dict[str, str]] = [
    {
        "dataset": "cachera2023 (betaxanthin)",
        "path": "torchcell/datasets/scerevisiae/cachera2023.py",
        "literal": 'media=Media(name="SC", state="solid", is_synthetic=True),',
        "recipe_key": "SC",
    },
    {
        "dataset": "ozaydin2013 (beta-carotene)",
        "path": "torchcell/datasets/scerevisiae/ozaydin2013.py",
        "literal": 'media=Media(name="SC-URA", state="solid", is_synthetic=True),',
        "recipe_key": "SC-URA",
    },
    {
        "dataset": "mulleder2016 (amino acids)",
        "path": "torchcell/datasets/scerevisiae/mulleder2016.py",
        "literal": 'media=Media(name="SM", state="solid", is_synthetic=True),',
        "recipe_key": "SM",
    },
    {
        "dataset": "ohya2005 (morphology)",
        "path": "torchcell/datasets/scerevisiae/ohya2005.py",
        "literal": 'media=Media(name="YPD", state="liquid", is_synthetic=False),',
        "recipe_key": "YPD",
    },
]

_MEDIA_KWARGS = re.compile(r"Media\((?P<args>[^)]*)\)")


def locate_literal(path: str, literal: str) -> int:
    """Line number of ``literal`` in ``path``, or fail loudly.

    No fallback: if the loader no longer contains the recorded line, the audit's claim
    about that loader is stale and the report must not be written.
    """
    full = osp.join(REPO_ROOT, path)
    with open(full, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip() == literal:
                return number
    raise RuntimeError(f"literal not found in {path}: {literal!r}")


def media_from_literal(literal: str) -> Media:
    """Rebuild the loader's ``Media`` from its own source line.

    Parsing the literal rather than retyping its fields is what makes the audited object
    the loader's object. The loaders build ``Media`` with three keyword arguments and
    nothing else, which is itself the finding, so the parser deliberately accepts only
    that form and raises on anything richer.
    """
    match = _MEDIA_KWARGS.search(literal)
    if match is None:
        raise RuntimeError(f"could not parse a Media(...) call from {literal!r}")
    kwargs: dict[str, Any] = {}
    for part in match.group("args").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        kwargs[key.strip()] = json.loads(
            value.strip().replace("True", "true").replace("False", "false")
        )
    return Media(**kwargs)


def coverage_record(
    label: str, media: Media, model: Any, index: ExchangeIndex, policy: UptakePolicy
) -> tuple[dict[str, Any], MediaBounds]:
    """Map one medium and describe the mapping plus the growth it supports."""
    bounds = media_to_bounds(media, model, policy=policy, index=index)

    channels: dict[str, int] = {}
    for resolution in bounds.resolutions:
        if resolution.match_channel is None:
            continue
        family = resolution.match_channel.split(":")[0]
        channels[family] = channels.get(family, 0) + 1

    with model:
        bounds.apply(model)
        solution = model.optimize()
        growth_status = solution.status
        # a non-optimal solve carries no growth rate; the solver's last objective value
        # on an infeasible problem is an artifact, so it is reported as absent
        growth_rate = (
            round(float(solution.objective_value), 6)
            if growth_status == "optimal"
            else None
        )

    return {
        "label": label,
        "media_name": media.name,
        "is_synthetic": media.is_synthetic,
        "state": media.state,
        "n_components": bounds.n_components,
        "n_resolved": bounds.n_resolved,
        "n_excluded_by_role": len(bounds.excluded_names),
        "n_unresolved": len(bounds.unresolved_names),
        "unresolved": [
            {
                "component": r.component_name,
                "role": r.role.value,
                "reason": r.reason,
                "candidates_tried": r.candidates_tried,
            }
            for r in bounds.resolutions
            if r.outcome == "unresolved"
        ],
        "excluded_by_role": [
            {"component": r.component_name, "role": r.role.value, "reason": r.reason}
            for r in bounds.resolutions
            if r.outcome == "excluded_by_role"
        ],
        "match_channels": channels,
        "n_bounds": len(bounds.bounds),
        "bounds": {
            exchange_id: {
                "metabolite": bound.metabolite_name,
                "uptake_bound": bound.uptake_bound,
                "unit": bound.unit,
            }
            for exchange_id, bound in sorted(bounds.bounds.items())
        },
        "fba": {"status": growth_status, "growth_rate_per_h": growth_rate},
    }, bounds


def annotation_channel_selftest(model: Any, index: ExchangeIndex) -> dict[str, Any]:
    """Round trip every SC component's model ChEBI id back through the annotation path.

    Exercises the identifier channel that the ontology's empty ``chebi_id`` fields leave
    dormant: take the metabolite each SC component resolved to, read its ChEBI id off the
    model, hand it back as ``Compound.chebi_id``, and require the same exchange.
    """
    from torchcell.datamodels.schema import Compound, MediaComponent, MediaComponentRole
    from torchcell.metabolism.media import resolve_component

    recipe = FBA_MEDIA["SC"]
    base = media_to_bounds(recipe, model, index=index)

    checked = 0
    agreed = 0
    no_chebi: list[str] = []
    for resolution in base.resolutions:
        if resolution.outcome != "resolved":
            continue
        exchange_id = resolution.exchange_ids[0]
        metabolite_id = index.metabolite_of[exchange_id][0]
        chebi = model.metabolites.get_by_id(metabolite_id).annotation.get("chebi")
        if chebi is None:
            no_chebi.append(resolution.component_name)
            continue
        chebi_value = chebi[0] if isinstance(chebi, list) else chebi
        probe = MediaComponent(
            compound=Compound(name="opaque probe", chebi_id=chebi_value),
            role=MediaComponentRole.other,
        )
        result = resolve_component(probe, index)
        checked += 1
        if result.outcome == "resolved" and result.exchange_ids == [exchange_id]:
            agreed += 1

    return {
        "description": "resolve each SC component's metabolite by its model ChEBI id "
        "alone, under a compound name the resolver cannot match",
        "n_checked": checked,
        "n_agreed": agreed,
        "components_without_model_chebi": no_chebi,
    }


def supplement_rate_comparison(
    model: Any, index: ExchangeIndex, glucose_rate: float = 10.0
) -> dict[str, Any]:
    """Sourced absolute supplement rate against the rescaled one our older code computes.

    ``experiments/007-kuzmin-tm/scripts/setup_media_conditions.py`` lines 71 and 124
    compute ``glucose_rate * 0.05``. Suthers' 0.165 is anchored to their DEFAULT 3.3
    carbon uptake and does not move. At the glucose rate those scripts pass, the ratio is
    the size of the divergence.
    """
    sourced = UptakePolicy(carbon_uptake=glucose_rate)
    rescaled = UptakePolicy(
        carbon_uptake=glucose_rate, supplement_uptake=glucose_rate * 0.05
    )

    rates: dict[str, Any] = {
        "glucose_rate": glucose_rate,
        "sourced_supplement_uptake": sourced.supplement_uptake,
        "rescaled_supplement_uptake": rescaled.supplement_uptake,
        "ratio_rescaled_over_sourced": round(
            rescaled.supplement_uptake / sourced.supplement_uptake, 4
        ),
        "growth_rate_per_h": {},
    }
    for key, media in FBA_MEDIA.items():
        entry: dict[str, float] = {}
        for label, policy in (("sourced", sourced), ("rescaled", rescaled)):
            bounds = media_to_bounds(media, model, policy=policy, index=index)
            with model:
                bounds.apply(model)
                entry[label] = round(float(model.optimize().objective_value), 6)
        rates["growth_rate_per_h"][key] = entry
    return rates


def ontology_library_coverage(
    model: Any, index: ExchangeIndex, policy: UptakePolicy
) -> dict[str, Any]:
    """Push the richest hand-written ontology media through the same mapping.

    ``torchcell/datamodels/media.py`` holds the fully sourced SGA and YPD recipes, and
    they carry exactly the things the four FBA recipes do not: agar, four selection
    agents, a ``composition_deferred`` commercial YNB line, an unexpanded amino-acid
    supplement powder, and two ``intrinsically_undefined`` digests. They are the test of
    whether the mapping stays honest when the ontology object is not GEM-shaped, and of
    whether the three-outcome record actually separates "cannot be represented" from
    "could not be found".
    """
    return {
        key: coverage_record(key, media, model, index, policy)[0]
        for key, media in MEDIA_LIBRARY.items()
    }


def main() -> None:
    """Audit the four media against yeast-GEM 9.0.2 and write the JSON report."""
    gem = YeastGEM()
    model = gem.model
    index = build_exchange_index(model)
    policy = UptakePolicy()

    report: dict[str, Any] = {
        "model": {
            "id": model.id,
            "yeast_gem_version": gem.version,
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_exchanges": len(model.exchanges),
            "n_exchanges_indexed": len(index.metabolite_of),
            "objective": str(model.objective.expression),
        },
        "policy": policy.model_dump(),
        "policy_note": (
            "supplement_uptake is Suthers 2020 sec2.5's ABSOLUTE 0.165 mmol/gDW/h, "
            "anchored to their default 3.3 carbon uptake; it does NOT rescale with "
            "carbon_uptake, which is where torchcell's older setup_media_conditions.py "
            "diverges from the source"
        ),
        "media": {},
        "pairwise_diff": {},
        "annotation_channel_selftest": annotation_channel_selftest(model, index),
        "supplement_rate_comparison": supplement_rate_comparison(model, index),
        "ontology_library_media": ontology_library_coverage(model, index, policy),
        "dataset_loader_media": [],
    }

    bounds_by_key: dict[str, Any] = {}
    for key, media in FBA_MEDIA.items():
        record, bounds = coverage_record(key, media, model, index, policy)
        report["media"][key] = record
        bounds_by_key[key] = bounds

    keys = list(FBA_MEDIA)
    for i, left_key in enumerate(keys):
        for right_key in keys[i + 1 :]:
            diff = diff_bounds(bounds_by_key[left_key], bounds_by_key[right_key])
            report["pairwise_diff"][f"{left_key}|{right_key}"] = {
                "n_differences": diff.n_differences,
                "only_in_left": {
                    k: {
                        "metabolite": bounds_by_key[left_key].bounds[k].metabolite_name,
                        "uptake_bound": v,
                    }
                    for k, v in diff.only_in_left.items()
                },
                "only_in_right": {
                    k: {
                        "metabolite": bounds_by_key[right_key]
                        .bounds[k]
                        .metabolite_name,
                        "uptake_bound": v,
                    }
                    for k, v in diff.only_in_right.items()
                },
                "differing": {
                    k: {"left": v[0], "right": v[1]} for k, v in diff.differing.items()
                },
            }

    for spec in DATASET_MEDIA_LITERALS:
        line = locate_literal(spec["path"], spec["literal"])
        emitted = media_from_literal(spec["literal"])
        record, _ = coverage_record(
            f"{spec['dataset']} as emitted", emitted, model, index, policy
        )
        recipe = FBA_MEDIA[spec["recipe_key"]]
        report["dataset_loader_media"].append(
            {
                "dataset": spec["dataset"],
                "source": f"{spec['path']}:{line}",
                "emitted_literal": spec["literal"],
                "emitted": emitted.model_dump(),
                "n_components_emitted": len(emitted.components),
                "coverage": record,
                "matching_recipe": {
                    "key": spec["recipe_key"],
                    "name": recipe.name,
                    "n_components": len(recipe.components),
                    "state_agrees": recipe.state == emitted.state,
                    "is_synthetic_agrees": recipe.is_synthetic == emitted.is_synthetic,
                },
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=False))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
