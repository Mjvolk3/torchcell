# experiments/036-dataset-fixes-before-kg-build/scripts/addition_family_census_correction.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.addition_family_census_correction]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/addition_family_census_correction
"""What the hand-written addition family was costing the construct-placement census.

``torchcell/verification/construct_placement.py`` named the addition leaves as a literal
pair, ``{"gene_addition", "heterologous_pathway"}``, while the schema union has FOUR
subclasses of ``GeneAdditionPerturbation``. Lane R measured that and left the fix here.
This script states both halves of the correction with numbers rather than with an
argument:

1. **The family.** The union's leaves, the four that subclass
   ``GeneAdditionPerturbation``, the four that declare ``placement``, and which of them
   the old literal pair named. Two independent properties of the class agreeing is what
   makes the derivation trustworthy rather than merely shorter.
2. **The census, old versus corrected, on the built dev stores that can hold one.**
   Each store's LMDB is read ONCE, read-only, and every addition perturbation is tallied
   under both families, so the row's numbers before and after are measured on the same
   bytes instead of being reasoned about. The stores read are the ones whose loader
   constructs an addition leaf, which is a property of the SOURCE and is passed as
   ``--store``; ``--all-stores`` scans every built store instead, which reads every
   record of every store (the Hale 2024 store alone is 8.7 million) and is therefore
   not the default. ``DEFAULT_STORES`` is the measured shortlist, and a store missing
   from it fails nothing: it is simply not in this report.

A store pickled under a SUPERSEDED schema closure cannot be unpickled by the current
schema at all (``Can't get attribute 'ReleasedGeneKind'``, a class the integrated branch
retired), so such a store is listed under ``unreadable`` with the attribute that is gone
rather than crashing the scan. That list is itself a measurement of why this branch's
schema-change record reads ``full_kg_rebuild``.

The dev tree is read, never the ``database/`` build tree, and nothing is written into it.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/addition_family_census_correction.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/addition_family_census_correction.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/addition_family_census_correction.py --store pooled_promoter_expression_vaishnav2022
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

import lmdb
from dotenv import load_dotenv

from torchcell.verification.construct_placement import (
    addition_perturbation_types,
    construct_placement_census,
)


class StoreClosureGone(RuntimeError):
    """A built store was pickled under a schema closure this tree no longer has."""


#: The family as it was written by hand, kept here so the correction is measurable and
#: not merely described. It is NOT imported from the module, because the module no
#: longer holds it.
OLD_FAMILY: frozenset[str] = frozenset({"gene_addition", "heterologous_pathway"})

#: The built dev stores whose loaders construct one of the four addition leaves,
#: measured by grepping ``torchcell/datasets/`` for the leaf class names. The two
#: formerly missed leaves come from ``vaishnav2022`` alone; the rest are here so the
#: unchanged half of the correction is measured too rather than assumed.
DEFAULT_STORES: tuple[str, ...] = (
    "betaxanthin_cachera2023",
    "carotenoid_ozaydin2013",
    "crispri_mormino2022",
    "pooled_promoter_expression_vaishnav2022",
)

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "addition_family_census_correction.json",
)


def _family_report() -> dict[str, Any]:
    """The union's addition leaves by two independent properties of the class."""
    from torchcell.datamodels.schema import (
        GeneAdditionPerturbation,
        GenePerturbationType,
    )

    leaves = get_args(GenePerturbationType)
    by_subclass = {
        str(leaf.model_fields["perturbation_type"].default)
        for leaf in leaves
        if issubclass(leaf, GeneAdditionPerturbation)
    }
    by_placement = {
        str(leaf.model_fields["perturbation_type"].default)
        for leaf in leaves
        if "placement" in leaf.model_fields
    }
    derived = addition_perturbation_types()
    return {
        "n_union_leaves": len(leaves),
        "subclasses_of_gene_addition": sorted(by_subclass),
        "leaves_declaring_placement": sorted(by_placement),
        "two_derivations_agree": by_subclass == by_placement,
        "derived_family": sorted(derived),
        "old_hand_written_family": sorted(OLD_FAMILY),
        "missed_by_the_old_family": sorted(derived - OLD_FAMILY),
        "named_by_the_old_family_only": sorted(OLD_FAMILY - derived),
    }


def _perturbations(lmdb_dir: Path) -> Iterator[dict[str, Any]]:
    """Every perturbation mapping of every record of one built store, read-only."""
    env = lmdb.open(
        str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = pickle.loads(value)  # noqa: S301  # our own built store
                experiment = record.get("experiment") or {}
                genotype = experiment.get("genotype") or {}
                yield from genotype.get("perturbations") or []
    finally:
        env.close()


def _store_report(name: str, lmdb_dir: Path) -> dict[str, Any] | None:
    """One store's addition tallies under both families, or None when it has none.

    Raises ``StoreClosureGone`` when the store was pickled under a schema closure the
    current tree no longer has, which is reported rather than swallowed.
    """
    derived = addition_perturbation_types()
    seen: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    try:
        for perturbation in _perturbations(lmdb_dir):
            tag = str(perturbation.get("perturbation_type"))
            if tag in derived:
                seen[tag] += 1
                kept.append(perturbation)
    except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as gone:
        raise StoreClosureGone(f"{name}: {gone}") from gone
    if not kept:
        return None
    old = construct_placement_census(
        [p for p in kept if str(p.get("perturbation_type")) in OLD_FAMILY]
    )
    new = construct_placement_census(kept)
    return {
        "store": name,
        "addition_types": dict(sorted(seen.items())),
        "n_additions_seen_by_the_old_family": old.n_additions,
        "n_additions_seen_by_the_derived_family": new.n_additions,
        "n_additions_the_old_family_skipped": new.n_additions - old.n_additions,
        "census_old": old.model_dump(),
        "census_corrected": new.model_dump(),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the family and the census correction over the built dev stores."""
    parser = argparse.ArgumentParser(prog="addition_family_census_correction")
    parser.add_argument("--store", action="append", dest="stores")
    parser.add_argument("--all-stores", action="store_true")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    if args.data_root is None:
        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
    else:
        data_root = args.data_root

    root = Path(data_root) / "data" / "torchcell"
    if args.stores:
        names = sorted(args.stores)
    elif args.all_stores:
        names = sorted(path.name for path in root.iterdir())
    else:
        names = sorted(DEFAULT_STORES)
    out: dict[str, Any] = {
        "family": _family_report(),
        "stores_read": names,
        "stores": [],
        "unreadable": [],
    }
    print(json.dumps(out["family"], indent=2, sort_keys=True))

    for name in names:
        lmdb_dir = root / name / "processed" / "lmdb"
        if not (lmdb_dir / "data.mdb").is_file():
            continue
        try:
            report = _store_report(name, lmdb_dir)
        except StoreClosureGone as gone:
            out["unreadable"].append({"store": name, "reason": str(gone)})
            continue
        if report is None:
            continue
        out["stores"].append(report)
        print(
            f"{name}: {report['n_additions_seen_by_the_old_family']} additions seen by "
            f"the old family, {report['n_additions_seen_by_the_derived_family']} by the "
            f"derived one ({report['n_additions_the_old_family_skipped']} skipped); "
            f"types {report['addition_types']}"
        )

    out["totals"] = {
        "n_stores_with_additions": len(out["stores"]),
        "n_stores_unreadable_under_the_current_closure": len(out["unreadable"]),
        "n_additions_seen_by_the_old_family": sum(
            s["n_additions_seen_by_the_old_family"] for s in out["stores"]
        ),
        "n_additions_seen_by_the_derived_family": sum(
            s["n_additions_seen_by_the_derived_family"] for s in out["stores"]
        ),
        "n_additions_the_old_family_skipped": sum(
            s["n_additions_the_old_family_skipped"] for s in out["stores"]
        ),
    }
    print(json.dumps(out["totals"], indent=2, sort_keys=True))
    for entry in out["unreadable"]:
        print(f"unreadable: {entry['reason']}")

    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
