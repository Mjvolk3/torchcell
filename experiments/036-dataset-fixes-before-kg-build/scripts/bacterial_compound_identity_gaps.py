# experiments/036-dataset-fixes-before-kg-build/scripts/bacterial_compound_identity_gaps.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.bacterial_compound_identity_gaps]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/bacterial_compound_identity_gaps
"""Enumerate every bacterial compound label with a typed InChIKey gap (#726).

#726 asks for a compound-identity curator pass over the bacterial condition labels, and
lists what was known at the time (Price 2018's 53, Tong 2020's 17, Goodall's
chloramphenicol, isoprenol, PRECISE-1K's 22 plus kanamycin, then D-alanine, L-malate and
alpha-pinene in the comments). That list is a running tally, not a measurement, so this
script measures the gap from the BUILT dev stores of every dataset in
``torchcell/knowledge_graphs/conf/kg_bacteria.yaml``.

What it reads, per store:

- the ``processed/interned`` env, which holds every deduplicated environment and
  reference, so one cheap scan covers every medium component and every dosed condition
  no matter how many records carry it;
- every record, for the stores whose PHENOTYPE carries a ``Compound`` (a product titer),
  since a phenotype is not interned.

A compound is a GAP when its ``inchikey`` is ``None``. The report splits those by whether
the compound is name-only (nothing but a name) or carries some other identifier, and names
the label, the stores it appears in, and whether the pinned table resolves it today.

It also re-checks the two things #726's comments warn about:

- **a row is not additive** if it renames a node in already-served records, so every gap
  label is looked up in the pinned table and its ``resolved_compound(...).name`` recorded;
  a label that already resolves to a DIFFERENT name is flagged ``renames_a_served_node``.
- the pinene row Niu 2019 needs: every spelling the issue lists is resolved against the
  pinned table, so the follow-up (#799) can read the state rather than re-measure it.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/bacterial_compound_identity_gaps.json``
and ``.../bacterial_compound_identity_gaps.csv`` (one row per gap label).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/bacterial_compound_identity_gaps.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/bacterial_compound_identity_gaps.py --dataset RbTnseqPrice2018EcoliDataset
"""

import argparse
import csv
import json
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
import yaml
from dotenv import load_dotenv

from torchcell.database.build_dataset_lmdb import (
    dataset_default_root,
    resolve_dataset_class,
)
from torchcell.datamodels.compound_identity import (
    _TABLE_SHA256,
    resolve_compound_identity,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
BACTERIA_CONF = "torchcell/knowledge_graphs/conf/kg_bacteria.yaml"
#: Every spelling of pinene #726's comment resolved, for the Niu 2019 follow-up (#799).
PINENE_SPELLINGS = (
    "pinene",
    "alpha-pinene",
    "alpha-Pinene",
    "(-)-alpha-pinene",
    "(+)-alpha-pinene",
)
#: The keys that make a dict a serialized ``Compound``.
_COMPOUND_KEYS = frozenset({"name", "inchikey", "inchi", "smiles", "pubchem_cid"})


def is_compound(obj: Any) -> bool:
    """True for a serialized ``Compound`` dict (it carries all five identity fields)."""
    return isinstance(obj, dict) and _COMPOUND_KEYS <= set(obj)


def walk_compounds(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every serialized ``Compound`` reachable from ``obj``."""
    if is_compound(obj):
        yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from walk_compounds(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            yield from walk_compounds(value)


def _scan_env(path: str) -> Iterator[dict[str, Any]]:
    """Stream every compound in one LMDB env, without collecting its records."""
    if not osp.isdir(path):
        return
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        for _key, value in txn.cursor():
            yield from walk_compounds(pickle.loads(value))
    env.close()


def interned_entries(root: str) -> int:
    """How many constants a store's ``interned`` env holds (0 when it has none)."""
    path = osp.join(root, "processed", "interned")
    if not osp.isdir(path):
        return 0
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        count = int(txn.stat()["entries"])
    env.close()
    return count


def carries_a_compound_phenotype(dataset_class: Any) -> bool:
    """Whether this dataset's PHENOTYPE can hold a ``Compound`` of its own.

    ``ProductTiterPhenotype.product`` is the only ``Compound`` on a phenotype; every
    other one hangs off ``Environment`` (a media component, a dosed small molecule, a
    solvent, a physical factor's agent, a media dropout), and ``Environment`` is
    interned. So for a dataset without a titer phenotype the interned env IS the whole
    compound surface and its records need not be read, which is what keeps this scan off
    the multi-gigabyte stores.
    """
    experiment_class = dataset_class.experiment_class.fget(
        dataset_class.__new__(dataset_class)
    )
    annotation = experiment_class.model_fields["phenotype"].annotation
    return "ProductTiterPhenotype" in str(annotation)


def store_compounds(root: str, read_records: bool) -> Iterator[dict[str, Any]]:
    """Stream the interned constants' compounds, and the records' too when asked.

    STREAMS rather than collecting: a million-record store holds millions of compound
    references, and accumulating them exhausts memory before the scan finishes.
    """
    yield from _scan_env(osp.join(root, "processed", "interned"))
    if read_records:
        yield from _scan_env(osp.join(root, "processed", "lmdb"))


#: Proposed TRIAGE of a gap label, by its own text. This is a grouping aid for the
#: curator pass, NOT a measurement: it says which directive a label probably wants, and
#: every assignment is a curation decision the owner still makes. The patterns are
#: deliberately literal so the grouping is reproducible.
TRIAGE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Not a compound at all: a placeholder naming a whole growth environment the
    # compendium did not release. These must never get a PubChem row.
    ("environment_placeholder", ("growth environment of",)),
    # A named preparation, not a molecule: a medium base, a trace-element stock, an
    # amino-acid mix, a vendor product. `undefined=` or `proprietary=`.
    (
        "undefined_or_proprietary_preparation",
        (
            "trace element",
            "trace metal",
            "metal trace",
            "medium",
            "medium salts",
            "base, PRECISE-1K formulation",
            "salts (Difco)",
            "_mix",
            "Durasyn",
            "Teknova",
        ),
    ),
)


def triage(label: str) -> str:
    """Which directive family ``label`` probably wants (a grouping aid, not a fact)."""
    for family, needles in TRIAGE_PATTERNS:
        if any(needle.lower() in label.lower() for needle in needles):
            return family
    return "single_substance_candidate"


def table_state(label: str) -> dict[str, Any]:
    """What the pinned table says about ``label`` today."""
    identity = resolve_compound_identity(label)
    return {
        "resolution_status": str(identity.status),
        "resolved_name": identity.name,
        "inchikey": identity.inchikey,
        "chebi_id": identity.chebi_id,
        "pubchem_cid": identity.pubchem_cid,
        "unresolved_reason": identity.unresolved_reason,
        "curated": identity.curated,
        "identified": identity.identified,
    }


def main() -> None:
    """Scan every bacterial dev store, then write the gap report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="limit the scan to these dataset class names (repeatable)",
    )
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    conf = yaml.safe_load(Path(BACTERIA_CONF).read_text())
    wanted = args.dataset or conf["datasets"]

    #: label -> {"stores": {...}, "shapes": [...]} for every compound with no inchikey.
    gaps: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"stores": set(), "name_only": True, "other_identifiers": {}}
    )
    scanned: dict[str, int] = {}
    scan_path: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    #: dataset class -> the unpickling error its store raised (a store to rebuild).
    unreadable: dict[str, str] = {}
    resolved_labels: set[str] = set()
    for class_name in wanted:
        dataset_class = resolve_dataset_class(class_name)
        root = osp.join(data_root, dataset_default_root(dataset_class))
        if not osp.isdir(osp.join(root, "processed")):
            missing.append(f"{class_name} (no dev store at {root})")
            continue
        interned = interned_entries(root)
        # Read the records only when they can hold a compound the interned constants
        # cannot: a titer phenotype, or a store that interned nothing at all.
        read_records = interned == 0 or carries_a_compound_phenotype(dataset_class)
        scan_path[class_name] = {
            "interned_entries": interned,
            "records_read": read_records,
        }
        seen = 0
        # Streamed with an explicit `next`, because a store built under older code can
        # hold a pickle naming a class the schema no longer has, which no current reader
        # can load. That is a FINDING (the store must be rebuilt before the KG build),
        # so it is recorded with its error and named in the report, never skipped
        # quietly, and the scan goes on to the next store.
        compounds = store_compounds(root, read_records)
        while True:
            try:
                compound = next(compounds)
            except StopIteration:
                break
            except AttributeError as error:
                unreadable[class_name] = f"{type(error).__name__}: {error}"
                print(f"  UNREADABLE {class_name}: {error}", flush=True)
                break
            seen += 1
            label = str(compound["name"])
            if compound["inchikey"] is not None:
                resolved_labels.add(label)
                continue
            record = gaps[label]
            record["stores"].add(class_name)
            others = {
                key: compound[key]
                for key in ("inchi", "smiles", "pubchem_cid", "chebi_id")
                if compound.get(key) is not None
            }
            if others:
                record["name_only"] = False
                record["other_identifiers"] |= others
        scanned[class_name] = seen
        print(f"  scanned {class_name}: {seen} compound references", flush=True)

    rows = []
    for label in sorted(gaps):
        state = table_state(label)
        rows.append(
            {
                "label": label,
                "stores": sorted(gaps[label]["stores"]),
                "name_only": gaps[label]["name_only"],
                "proposed_triage": triage(label),
                "other_identifiers": gaps[label]["other_identifiers"],
                **state,
                # A label the table ALREADY resolves to a different name would rename
                # this node the moment the loader stops gapping it (#726 comment).
                "renames_a_served_node": (
                    state["resolved_name"] is not None
                    and state["resolved_name"] != label
                ),
            }
        )

    result: dict[str, Any] = {
        "table_sha256": _TABLE_SHA256,
        "datasets_requested": len(wanted),
        "datasets_scanned": len(scanned),
        "datasets_not_scanned": missing,
        "datasets_whose_store_cannot_be_read": unreadable,
        "compound_objects_scanned": sum(scanned.values()),
        "distinct_resolved_labels": len(resolved_labels),
        "gap_labels": len(rows),
        "gap_labels_name_only": sum(1 for row in rows if row["name_only"]),
        "gap_labels_by_proposed_triage": dict(
            sorted(Counter(row["proposed_triage"] for row in rows).items())
        ),
        "gap_labels_that_would_rename_a_served_node": [
            row["label"] for row in rows if row["renames_a_served_node"]
        ],
        "gaps": rows,
        "pinene_spellings": {
            spelling: table_state(spelling) for spelling in PINENE_SPELLINGS
        },
        "per_dataset_compound_objects": dict(sorted(scanned.items())),
        "per_dataset_scan_path": dict(sorted(scan_path.items())),
    }

    os.makedirs(RESULTS, exist_ok=True)
    json_path = osp.join(RESULTS, "bacterial_compound_identity_gaps.json")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    csv_path = osp.join(RESULTS, "bacterial_compound_identity_gaps.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "proposed_triage",
                "name_only",
                "resolution_status",
                "resolved_name",
                "stores",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    row["proposed_triage"],
                    row["name_only"],
                    row["resolution_status"],
                    row["resolved_name"] or "",
                    ";".join(row["stores"]),
                ]
            )
    print(json.dumps({k: v for k, v in result.items() if k not in {"gaps"}}, indent=2))
    for row in rows:
        print(
            f"  [{row['resolution_status']}] [{row['proposed_triage']}] "
            f"{row['label']}: {row['stores']}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
