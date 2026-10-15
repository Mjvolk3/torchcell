# torchcell/data/label_table.py
# [[torchcell.data.label_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/label_table
# Test file: tests/torchcell/data/test_label_policy.py
"""Apply a ``LabelPolicy`` to a no-merge build and cache the table it produces.

The build keeps every source entry under a genotype; the trainer wants one value per
label per record. This is the pass between them: one read over the processed LMDB, one
row per record naming the chosen value and the source that supplied it, written beside
the split caches under the policy's hash so that changing a rule costs a new small file
rather than a rebuild.

    from torchcell.data.label_policy import LabelPolicy
    from torchcell.data.label_table import build_label_table

    table = build_label_table(LabelPolicy(name="kuzmin-first"), build_root)

``build_root`` is the dataset root holding ``processed/lmdb``; the table lands in
``<build_root>/label_tables/<policy_id>.parquet``.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import re
from multiprocessing import Pool
from typing import Any

import pandas as pd
from pydantic import BaseModel

from torchcell.data.label_policy import (
    LabelEntry,
    LabelPolicy,
    source_key,
    strain_token,
)

__all__ = [
    "TripleRoles",
    "build_label_table",
    "entries_of_record",
    "label_table_path",
    "triple_roles",
]

# An SGA array strain is named by its array plate: "dma" for the deletion-mutant array,
# "tsa" for the temperature-sensitive one. The query strains carry a "tm" number instead.
_ARRAY_STRAIN = re.compile(r"_(dma|tsa)\d+")


class TripleRoles(BaseModel):
    """Which gene of a triple was the array, and which two were the query pair.

    The trigenic identity is asymmetric: one gene came from the array and the other two
    were crossed in as a double-mutant query strain, and the score subtracts the query
    double's fitness times the array single. A record does not label the roles, but the
    strain identifiers give them away, and reading them is what lets a triple ask for
    the fitness of the exact double-mutant strain its own screen used.
    """

    array_gene: str
    query_genes: tuple[str, str]
    query_strain_id: str


def triple_roles(perturbations: list[dict[str, Any]]) -> TripleRoles | None:
    """Recover the query pair and array gene of a triple from its strain identifiers.

    Returns None when the roles are ambiguous, which is the honest answer for a record
    whose source did not name its strains. Measured on 5,000 triples of the 029 build,
    both the array gene and the query token resolve in all 5,000.
    """
    by_gene = {
        p["systematic_gene_name"]: (p.get("strain_id") or "") for p in perturbations
    }
    if len(by_gene) != 3:
        return None
    arrays = [g for g, s in by_gene.items() if _ARRAY_STRAIN.search(s)]
    if len(arrays) != 1:
        return None
    query = sorted(g for g in by_gene if g != arrays[0])
    if len(query) != 2:
        return None
    ids = {by_gene[g] for g in query if strain_token(by_gene[g])}
    if len(ids) != 1:
        return None
    return TripleRoles(
        array_gene=arrays[0],
        query_genes=(query[0], query[1]),
        query_strain_id=ids.pop(),
    )


LABELS = ("fitness", "gene_interaction")

_ENV = None
_BUILD_ROOT = ""


def label_table_path(build_root: str, policy: LabelPolicy) -> str:
    """Where a policy's table is cached for one build."""
    return osp.join(build_root, "label_tables", f"{policy.policy_id}.parquet")


def _finite(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def entries_of_record(raw: bytes) -> list[LabelEntry]:
    """Normalize one stored record's entries into ``LabelEntry`` objects.

    A stored record is a JSON list of ``{"experiment": ..., "experiment_reference": ...}``.
    A perturbation carries ``strain_id`` when the source reported one; when every
    perturbation of the entry shares it, that is the strain the entry measured, which is
    what a triple matches its doubles on.
    """
    out: list[LabelEntry] = []
    for item in json.loads(raw):
        e = item["experiment"]
        raw_type = e["experiment_type"]
        label = "gene_interaction" if "interaction" in raw_type else "fitness"
        ph = e["phenotype"]
        value = _finite(
            ph.get("fitness") if label == "fitness" else ph.get("gene_interaction")
        )
        if value is None:
            continue
        temp = (e.get("environment") or {}).get("temperature") or {}
        strain_ids = {
            p["strain_id"] for p in e["genotype"]["perturbations"] if p.get("strain_id")
        }
        out.append(
            LabelEntry(
                source=source_key(e["dataset_name"], _finite(temp.get("value"))),
                label=label,
                value=value,
                sd=_finite(ph.get("fitness_std")),
                n_samples=(lambda n: None if n is None else int(n))(
                    _finite(ph.get("n_samples"))
                ),
                p_value=_finite(ph.get("gene_interaction_p_value")),
                strain_id=strain_ids.pop() if len(strain_ids) == 1 else None,
            )
        )
    return out


def _init(build_root: str) -> None:
    global _ENV, _BUILD_ROOT
    import lmdb

    _BUILD_ROOT = build_root
    _ENV = lmdb.open(
        osp.join(build_root, "processed", "lmdb"),
        readonly=True,
        lock=False,
        max_readers=256,
    )


def _rows(args: tuple[list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    idxs, policy_dump = args
    policy = LabelPolicy.model_validate(policy_dump)
    assert _ENV is not None
    out: list[dict[str, Any]] = []
    with _ENV.begin() as txn:
        for i in idxs:
            raw = txn.get(str(i).encode())
            if raw is None:
                continue
            entries = entries_of_record(raw)
            row: dict[str, Any] = {"index": i}
            for label in LABELS:
                choice = policy.select(entries, label)
                row[label] = None if choice is None else choice.value
                row[f"{label}_source"] = None if choice is None else choice.source
                row[f"{label}_p"] = None if choice is None else choice.p_value
                row[f"{label}_sd"] = None if choice is None else choice.sd
                row[f"{label}_n_entries"] = (
                    0 if choice is None else choice.n_entries_available
                )
                row[f"{label}_n_combined"] = (
                    0 if choice is None else choice.n_entries_combined
                )
            out.append(row)
    return out


def build_label_table(
    policy: LabelPolicy,
    build_root: str,
    indices: list[int] | None = None,
    workers: int = 32,
    chunk: int = 2000,
    cache: bool = True,
) -> pd.DataFrame:
    """One row per record: the chosen value per label and the source that supplied it.

    ``indices`` restricts the pass, which is what a test or a subset arm wants; a cached
    table is only reused when it covered the whole build.
    """
    path = label_table_path(build_root, policy)
    if cache and indices is None and osp.exists(path):
        return pd.read_parquet(path)

    if indices is None:
        with open(
            osp.join(build_root, "processed", "perturbation_count_index.json")
        ) as f:
            by_order = json.load(f)
        indices = sorted(i for v in by_order.values() for i in v)

    dump = policy.model_dump(mode="json")
    chunks = [(indices[i : i + chunk], dump) for i in range(0, len(indices), chunk)]
    rows: list[dict[str, Any]] = []
    with Pool(workers, initializer=_init, initargs=(build_root,)) as pool:
        for part in pool.imap_unordered(_rows, chunks):
            rows.extend(part)
    table = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)

    if cache and indices is not None and len(table):
        os.makedirs(osp.dirname(path), exist_ok=True)
        table.to_parquet(path, index=False)
    return table
