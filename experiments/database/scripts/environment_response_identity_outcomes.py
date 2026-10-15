# experiments/database/scripts/environment_response_identity_outcomes.py
# [[experiments.database.scripts.environment_response_identity_outcomes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/environment_response_identity_outcomes
"""Do the environment-response identity projections change an outcome on real stores?

Lane W of the ontology unification. Lane R measured that
``environment_response._genotype_signature._identity`` reads 17 of the 92 fields the 38
genotype leaves declare beyond ``GenePerturbation``, and that the three identity sites
disagree, and deferred the question of whether that costs anything. This script answers it
by re-keying every store in ``ENVIRONMENT_RESPONSE_DATASETS`` under bracketing keys and
comparing the duplicate counts, read-only, in one streaming pass per store:

* ``head``   -- the seven-element condition signature the rule carried before this lane,
  with the shipped genotype signature. Recomputed here rather than imported, so the table
  is reproducible after the fix as well as before it.
* ``shipped`` -- :func:`torchcell.verification.environment_response._pair_key` as it
  stands now.
* ``genotype_base`` -- the shipped key with EVERY beyond-base genotype field dropped, so
  the strain is only ``(systematic_gene_name, perturbation_type, perturbed_gene_name)``.
* ``genotype_full`` -- the shipped key with the WHOLE stored perturbation dump, so no
  stored genotype field can merge two records.
* ``environment_full`` -- the head key with the WHOLE stored environment, so no stored
  environment field can merge two records.

The bracketing is what makes the answer complete: a key that reads MORE fields can only
split a group, never merge two, so ``head - genotype_full`` is the number of records that
collapse under the shipped key although some unread GENOTYPE field distinguishes them, and
``head - environment_full`` is the same number for the environment axis. Zero means no
field classification could change this store's accounting.

Three dev stores of this family were built before the unification's renames and raise
``AttributeError: ReleasedGeneKind`` under ``pickle``; they are read through an alias map
that resolves the old name to the renamed class and reports every substitution, so a stale
store is named in the output instead of being skipped. The KG 4.0 full rebuild remakes
them ([[database.kg-4-pre-build-verification]]).

Results: `experiments/database/results/environment_response_identity_outcomes.json`,
written up in [[torchcell.verification.environment-response-identity]].
"""

from __future__ import annotations

import argparse
import io
import json
import os.path as osp
import pickle
from collections import Counter
from hashlib import blake2b
from typing import Any

import lmdb

import torchcell.datamodels.schema as schema
from torchcell.verification.environment_response import (
    _condition_signature,
    _genotype_signature,
    _pair_key,
    _strain_background_name,
    _study_key,
)
from torchcell.verification.runners import ENVIRONMENT_RESPONSE_DATASETS

#: Old class names three dev stores of this family were pickled under, and the names the
#: ontology unification gave them. Read-only: the payload is dicts plus StrEnum values, so
#: resolving the old name reads the stored bytes without changing them.
RENAMED_CLASSES: dict[str, str] = {
    "ReleasedGeneKind": "ReleasedKind",
    "ReleasedGeneNamePerturbation": "ReleasedIdentifierPerturbation",
    "BacterialVariantType": "VariantType",
}

BASE_PERTURBATION_FIELDS = (
    "systematic_gene_name",
    "perturbation_type",
    "perturbed_gene_name",
)

KEYS = ("head", "shipped", "genotype_base", "genotype_full", "environment_full")


class _RenameTolerantUnpickler(pickle.Unpickler):
    """Unpickler that resolves a pre-unification class name to its renamed class."""

    def __init__(self, payload: bytes, substitutions: Counter[str]) -> None:
        super().__init__(io.BytesIO(payload))
        self._substitutions = substitutions

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError):
            renamed = RENAMED_CLASSES.get(name)
            if renamed is None:
                raise
            resolved = getattr(schema, renamed)
            self._substitutions[f"{name} -> {renamed}"] += 1
            return resolved


def _resolve_interned(obj: Any, interned: dict[str, Any]) -> Any:
    """Splice interned sub-objects back in, as ``experiment_dataset`` does."""
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if ref is not None:
            return interned[ref]
        return {key: _resolve_interned(value, interned) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_interned(value, interned) for value in obj]
    return obj


def stream_records(abs_root: str, substitutions: Counter[str]) -> Any:
    """Yield every record of a store, read-only, through the rename-tolerant unpickler."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(abs_root, "processed", "interned")
    if osp.isdir(interned_dir):
        env = lmdb.open(interned_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = _RenameTolerantUnpickler(
                    value, substitutions
                ).load()
        env.close()
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            for _, value in txn.cursor():
                record = _RenameTolerantUnpickler(value, substitutions).load()
                yield _resolve_interned(record, interned)
    finally:
        env.close()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _digest(payload: str) -> bytes:
    return blake2b(payload.encode(), digest_size=12).digest()


def head_condition_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """The seven-element condition signature the rule carried before lane W."""
    return _condition_signature(experiment)[:7]


def _genotype(
    experiment: dict[str, Any], background: frozenset[str], fields: Any
) -> tuple[Any, ...]:
    """The strain signature over ``fields``: the base triple, or the whole dump."""
    perturbations = [
        p
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") not in background
    ]
    if fields is None:
        return tuple(sorted(_canonical(p) for p in perturbations))
    return tuple(sorted(tuple(p.get(f) for f in fields) for p in perturbations))


def measure(name: str, data_root: str) -> dict[str, Any]:
    """Duplicate counts under the five keys, plus the environment-axis blind spots."""
    spec = ENVIRONMENT_RESPONSE_DATASETS[name]
    background: frozenset[str] = spec.get("background_genes", frozenset())
    abs_root = osp.join(data_root, spec["root"])
    substitutions: Counter[str] = Counter()

    seen: dict[str, set[bytes]] = {key: set() for key in KEYS}
    # one condition signature -> the distinct whole environments stored under it
    blind: dict[bytes, dict[bytes, str]] = {}
    n_records = 0
    for record in stream_records(abs_root, substitutions):
        experiment = record["experiment"]
        n_records += 1
        head = head_condition_signature(experiment)
        study = _study_key(record)
        strain_background = _strain_background_name(record)
        shipped_genotype = _genotype_signature(experiment, background)
        tail = (study, strain_background)
        environment = _canonical(experiment["environment"])
        seen["head"].add(_digest(repr((tail, shipped_genotype, head))))
        seen["shipped"].add(_digest(repr(_pair_key(record, background))))
        seen["genotype_base"].add(
            _digest(
                repr(
                    (
                        tail,
                        _genotype(experiment, background, BASE_PERTURBATION_FIELDS),
                        head,
                    )
                )
            )
        )
        seen["genotype_full"].add(
            _digest(repr((tail, _genotype(experiment, background, None), head)))
        )
        seen["environment_full"].add(
            _digest(repr((tail, shipped_genotype)) + environment + str(head[6]))
        )
        signature = _digest(repr((head, str(head[6]))))
        bucket = blind.setdefault(signature, {})
        if len(bucket) < 6:
            bucket.setdefault(_digest(environment), environment)

    paths: Counter[str] = Counter()
    n_blind = 0
    for bucket in blind.values():
        if len(bucket) < 2:
            continue
        n_blind += 1
        flattened = [_flatten(json.loads(dump)) for dump in bucket.values()]
        for path in set().union(*(set(flat) for flat in flattened)):
            if len({_canonical(flat.get(path)) for flat in flattened}) > 1:
                paths[path] += 1
    return {
        "dataset": name,
        "n_records": n_records,
        "declared_expected_duplicated": spec.get("expected_duplicated_pairs", 0),
        "duplicated": {key: n_records - len(seen[key]) for key in KEYS},
        "n_condition_signatures": len(blind),
        "n_condition_signatures_over_several_environments": n_blind,
        "blind_environment_paths": dict(paths.most_common(20)),
        "renamed_classes_read": dict(substitutions),
    }


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """``path -> scalar`` for every leaf of a stored object."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            flat |= _flatten(value, f"{prefix}[{index}]")
    else:
        flat[prefix] = obj
    return flat


#: The beyond-base genotype fields each of the three identity sites reads, read off their
#: source on 2026.10.15. A nested field is named by the object it is read from, since that
#: is what the projection dispatches on.
SITE_READ_FIELDS: dict[str, tuple[str, ...]] = {
    "environment_response._genotype_signature": (
        "allele_name",
        "cassette",
        "collection",
        "constructed_orf",
        "construction",
        "degron",
        "donor_sequence",
        "crispr",
        "insertion_site",
        "perturbation_type",
        "released_gene_id",
        "released_kind",
        "strain_label",
        "tag",
        "terminus",
    ),
    "fitness._genotype_signature": (
        "allele_name",
        "crispr",
        "perturbation_type",
        "released_gene_id",
        "strain_id",
    ),
    "metabolite._genotype_signature": (
        "construct_name",
        "perturbation_type",
        "variant",
    ),
}


def field_table() -> dict[str, Any]:
    """Which of the genotype leaves' beyond-base fields each identity site reads."""
    from typing import get_args

    from torchcell.datamodels.schema import GenePerturbation, GenePerturbationType

    leaves = get_args(GenePerturbationType)
    base = set(GenePerturbation.model_fields)
    beyond: dict[str, list[str]] = {}
    for leaf in leaves:
        for field in set(leaf.model_fields) - base:
            beyond.setdefault(field, []).append(leaf.__name__)
    read_by = {
        field: sorted(
            site for site, fields in SITE_READ_FIELDS.items() if field in fields
        )
        for field in beyond
    }
    unread = sorted(field for field, sites in read_by.items() if not sites)
    return {
        "n_leaves": len(leaves),
        "base_fields": sorted(base),
        "n_fields_beyond_base": len(beyond),
        "n_read_by_some_site": len(beyond) - len(unread),
        "n_read_by_no_site": len(unread),
        "read_by": {field: sites for field, sites in sorted(read_by.items()) if sites},
        "read_by_no_site": {field: sorted(beyond[field]) for field in unread},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="an ENVIRONMENT_RESPONSE_DATASETS key; repeatable, default every store",
    )
    parser.add_argument(
        "--field-table",
        action="store_true",
        help="print the per-site field classification instead of reading any store",
    )
    parser.add_argument("--data-root", default=osp.expandvars("$DATA_ROOT"))
    parser.add_argument(
        "--out",
        default=osp.join(
            "experiments",
            "database",
            "results",
            "environment_response_identity_outcomes.json",
        ),
    )
    args = parser.parse_args()
    if args.field_table:
        table = field_table()
        print(
            f"{table['n_leaves']} genotype leaves, "
            f"{len(table['base_fields'])} base fields, "
            f"{table['n_fields_beyond_base']} fields beyond the base: "
            f"{table['n_read_by_some_site']} read by some identity site, "
            f"{table['n_read_by_no_site']} read by none"
        )
        for field, sites in table["read_by"].items():
            print(f"  {field:22s} {', '.join(site.split('.')[0] for site in sites)}")
        with open(args.out, "w") as handle:
            json.dump(table, handle, indent=2)
        print(f"-> wrote {args.out}")
        return
    names = args.dataset or list(ENVIRONMENT_RESPONSE_DATASETS)
    rows = []
    for name in names:
        row = measure(name, args.data_root)
        rows.append(row)
        print(
            f"{row['dataset']:34s} n={row['n_records']:9d} "
            + " ".join(f"{key}={row['duplicated'][key]}" for key in KEYS)
            + f" declared={row['declared_expected_duplicated']}"
            + f" blind={row['n_condition_signatures_over_several_environments']}"
        )
    with open(args.out, "w") as handle:
        json.dump({"data_root": args.data_root, "stores": rows}, handle, indent=2)
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
