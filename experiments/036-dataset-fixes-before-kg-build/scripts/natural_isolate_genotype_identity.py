# experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.natural_isolate_genotype_identity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py
"""Do the natural-isolate stores build the genotypes ``natural_isolate`` builds (#988)?

Every number in the 2026.10.12 section of ``[[torchcell.sequence.natural_isolate]]``
about genotype identity comes from this script. Two modes:

``before`` (run on the stores the loaders built before the shared module): for every
isolate of every store, is the stored genotype dump EQUAL to
``NaturalIsolateGenotypes.base_perturbations`` (sequence variants, natural presences,
natural absences: what the loaders built before #987 / #988)? Also: do two stores that
share an isolate store the same genotype? Writes ``natural_isolate_genotype_identity_before.json``.

``after`` (run on the rebuilt stores, whose predecessors ``build_dataset_lmdb
--retire-existing`` kept as ``processed.superseded.<stamp>``): is every stored genotype
equal to ``NaturalIsolateGenotypes.genotype``, does every shared isolate carry one
genotype across stores, and what EXACTLY differs between an isolate's old and new
genotype (perturbations added and removed, by type)? Writes
``natural_isolate_genotype_identity_after.json``.

The content id reported is the sha256 of the sort-keyed JSON of the python-mode dump,
the same bytes for a stored dict and a rebuilt one.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py before
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import pickle
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Any

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.schema import Genotype
from torchcell.sequence.natural_isolate import (
    NaturalIsolateGenotypes,
    genotype_isolate,
    link_genotype_files,
    panel_isolates,
)
from torchcell.verification.runners import _load_interned

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
STORES = (
    "caudal_pantranscriptome2024",
    "isolate_growth_peter2018",
    "proteome_muenzner2024",
    "protein_turnover_muenzner2024",
    "doubling_time_dechiara2022",
    "cell_yield_dechiara2022",
    "life_cycle_dechiara2022",
)


def content_id(genotype: Mapping[str, Any]) -> str:
    """sha256 of the sort-keyed JSON of a python-mode genotype dump."""
    return hashlib.sha256(
        json.dumps(genotype, sort_keys=True, default=str).encode()
    ).hexdigest()


def old_isolate(genotype: Mapping[str, Any]) -> str:
    """The isolate a pre-#988 genotype names (every perturbation carried strain_id)."""
    strains = {str(p["strain_id"]) for p in genotype["perturbations"]}
    if len(strains) != 1:
        raise ValueError(f"a genotype names {sorted(strains)[:3]}")
    return next(iter(strains))


def store_genotypes(processed: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """``(isolate, genotype dump)`` once per isolate of the store under ``processed``.

    ``processed`` is a store's ``processed`` directory or a superseded copy of it; the
    interned env beside the LMDB is resolved. An isolate whose records carry two
    different genotypes raises.
    """
    root = osp.dirname(processed)
    interned = _load_interned_at(processed)

    seen: dict[str, str] = {}
    refs: set[str] = set()
    env = lmdb.open(osp.join(processed, "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for _, value in txn.cursor():
            raw = pickle.loads(value)
            pointer = raw["experiment"]["genotype"]
            # An interned genotype is one stored object shared by all of an isolate's
            # records: read and hash it once (its content id is its interning key).
            if isinstance(pointer, dict) and "$ref" in pointer:
                if pointer["$ref"] in refs:
                    continue
                refs.add(pointer["$ref"])
            record = resolve_interned(raw, interned)
            genotype = dict(record["experiment"]["genotype"])
            isolate = genotype_isolate(genotype) or old_isolate(genotype)
            ident = content_id(genotype)
            if isolate in seen:
                if seen[isolate] != ident:
                    raise ValueError(f"{root}: {isolate} carries two genotypes")
                continue
            seen[isolate] = ident
            yield isolate, genotype
    env.close()


def _load_interned_at(processed: str) -> dict[str, Any]:
    """The interned table of a (possibly superseded) ``processed`` directory."""
    if osp.basename(processed) == "processed":
        return _load_interned(osp.dirname(processed))
    path = osp.join(processed, "interned")
    if not osp.isdir(path):
        return {}

    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        table = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
    env.close()
    return table


class StoreIdentity(BaseModel):
    """One store's genotypes against the shared module."""

    model_config = ConfigDict(extra="forbid")

    store: str
    processed: str
    n_isolates: int
    n_identical: int
    mismatched: list[str]


class _Ids:
    """Content ids of the shared module's genotypes, computed once per isolate."""

    def __init__(self, genotypes: NaturalIsolateGenotypes, full: bool) -> None:
        self.genotypes = genotypes
        self.full = full
        self.ids: dict[str, str] = {}

    def __getitem__(self, isolate: str) -> str:
        if isolate not in self.ids:
            genotype = (
                self.genotypes.genotype(isolate)
                if self.full
                else Genotype(perturbations=self.genotypes.base_perturbations(isolate))
            )
            self.ids[isolate] = content_id(genotype.model_dump())
        return self.ids[isolate]


def _builder(data_root: str, workdir: str) -> NaturalIsolateGenotypes:
    """The shared module over all 1,011 genome isolates (allele diff in ``workdir``)."""
    raw = osp.join(workdir, "raw")
    os.makedirs(raw, exist_ok=True)
    link_genotype_files(raw, data_root)
    return NaturalIsolateGenotypes(raw, workdir, panel_isolates(raw), data_root)


def _compare(
    stores: dict[str, str], ids: _Ids
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, set[str]] = {}
    n_stores: Counter[str] = Counter()
    for store, processed in stores.items():
        bad: list[str] = []
        n = 0
        for isolate, genotype in store_genotypes(processed):
            n += 1
            ident = content_id(genotype)
            seen.setdefault(isolate, set()).add(ident)
            n_stores[isolate] += 1
            if ident != ids[isolate]:
                bad.append(isolate)
        rows.append(
            StoreIdentity(
                store=store,
                processed=processed,
                n_isolates=n,
                n_identical=n - len(bad),
                mismatched=sorted(bad)[:20],
            ).model_dump()
        )
    return rows, seen, dict(n_stores)


def _cross_store(seen: dict[str, set[str]], n_stores: dict[str, int]) -> dict[str, int]:
    shared = [iso for iso, n in n_stores.items() if n > 1]
    return {
        "n_isolates_any_store": len(seen),
        "n_isolates_in_two_or_more_stores": len(shared),
        "shared_isolates_with_one_content_id": sum(
            1 for iso in shared if len(seen[iso]) == 1
        ),
    }


def before(data_root: str) -> dict[str, Any]:
    """Stored genotypes vs ``base_perturbations``, and agreement across stores."""
    stores = {s: osp.join(data_root, "data/torchcell", s, "processed") for s in STORES}
    with tempfile.TemporaryDirectory() as workdir:
        ids = _Ids(_builder(data_root, workdir), full=False)
        rows, seen, n_stores = _compare(stores, ids)
    return {"stores": rows, **_cross_store(seen, n_stores)}


def _perturbation_types(processed: str) -> dict[str, dict[str, str]]:
    """``isolate -> {perturbation content id: type}`` for one store."""
    return {
        isolate: {
            content_id(p): str(p["perturbation_type"])
            for p in genotype["perturbations"]
        }
        for isolate, genotype in store_genotypes(processed)
    }


def after(data_root: str) -> dict[str, Any]:
    """Rebuilt genotypes vs ``genotype``, across stores, and old vs new by type."""
    stores = {s: osp.join(data_root, "data/torchcell", s, "processed") for s in STORES}
    with tempfile.TemporaryDirectory() as workdir:
        genotypes = _builder(data_root, workdir)
        ids = _Ids(genotypes, full=True)
        rows, seen, n_stores = _compare(stores, ids)
        ledger = genotypes.ledger().model_dump(mode="json")
    for row in rows:
        root = osp.dirname(row["processed"])
        superseded = sorted(glob.glob(osp.join(root, "processed.superseded.*")))
        added: Counter[str] = Counter()
        removed: Counter[str] = Counter()
        n_compared = 0
        if superseded:
            old = _perturbation_types(superseded[-1])
            new = _perturbation_types(row["processed"])
            for isolate, after_types in new.items():
                if isolate not in old:
                    continue
                n_compared += 1
                before_types = old[isolate]
                added.update(
                    after_types[k] for k in after_types.keys() - before_types.keys()
                )
                removed.update(
                    before_types[k] for k in before_types.keys() - after_types.keys()
                )
        row["superseded"] = superseded[-1] if superseded else None
        row["n_isolates_compared_with_superseded"] = n_compared
        row["perturbations_added_by_type"] = dict(sorted(added.items()))
        row["perturbations_removed_by_type"] = dict(sorted(removed.items()))
    return {
        "stores": rows,
        **_cross_store(seen, n_stores),
        "ledger_over_the_1011_genome_isolates": ledger,
    }


def main() -> None:
    """Run one mode and write its JSON beside the other results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "after"])
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    result = before(data_root) if args.mode == "before" else after(data_root)
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, f"natural_isolate_genotype_identity_{args.mode}.json")
    with open(path, "w") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True)[:6000])
    print(path)


if __name__ == "__main__":
    main()
