# experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_store_reach.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_curator_726_860_store_reach]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_store_reach
r"""Which built dev stores the #726/#860 curator rows reach, measured from the stores.

A compound row changes nothing in a store until the store is rebuilt: the records pickle
the ``Compound`` the resolver returned at build time. So the rebuild set of the curator
pass is every mapped store holding a compound whose name is one of the curated labels
(casefolded, the resolver's lookup key) AND still carrying no InChIKey where the row now
gives one, or a name the row now spells differently. The committed gap inventory found
the labels in the bacterial stores only; this scan reads EVERY mapped store, yeast
included, so a yeast loader passing the same label (``arginine``, ``water``) is seen
rather than assumed absent.

For each store it reports, per curated label it holds: the name the records carry, how
many compound references carry it, how many of those have an InChIKey, and what the
pinned table now returns for that name. A store is ``stale`` when any held compound
differs from what the resolver would build today: its name, its InChIKey, a PubChem CID
the row now fills, or the reason of its ``inchikey`` gap (a mixture row makes a
recoverable gap terminal). The CID and gap-reason checks were added after the "before"
run (whose 40-store set they would not have grown: every store holding a mixture label is
in it), so the "after" run applies the stricter test. Run before the rebuild for the
rebuild set and after it to show the set is empty.

Reads each store the way ``bacterial_compound_identity_gaps.py`` does: the
``processed/interned`` env (every deduplicated environment), plus the records for a
dataset whose phenotype can hold a ``Compound`` (a titer) or that interned nothing.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860_store_reach_<tag>.json``
and, with ``--classes-file``, the stale class names one per line (the
``gilahyper_build_dataset_lmdbs_array.slurm`` CLASSES_FILE).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_store_reach.py --tag before \
        --classes-file /scratch/projects/torchcell/database/slurm/classes_726_860.txt
"""

import argparse
import importlib.util
import json
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from torchcell.database.build_dataset_lmdb import dataset_default_root
from torchcell.datamodels.compound_identity import (
    _TABLE_SHA256,
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.compound_identity_curate import read_name_list
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
INPUT_LIST = Path(
    "torchcell/datamodels/compound_identity_inputs/bacterial_conditions.txt"
)
GAP_SCRIPT = Path(
    "experiments/036-dataset-fixes-before-kg-build/scripts/"
    "bacterial_compound_identity_gaps.py"
)


def _gap_module() -> Any:
    """The gap inventory's store readers, loaded from its file (one reader, not two)."""
    spec = importlib.util.spec_from_file_location("compound_gaps", GAP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def curated_keys() -> dict[str, str]:
    """Casefolded lookup key -> input label, for every line of the curated list."""
    return {d.label.strip().lower(): d.label for d in read_name_list(INPUT_LIST)}


def main() -> None:
    """Scan every mapped dev store for the curated labels and write the reach report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="'before' or 'after' the rebuild")
    parser.add_argument("--classes-file", type=Path, default=None)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    gaps = _gap_module()
    keys = curated_keys()

    stores: dict[str, Any] = {}
    unreadable: dict[str, str] = {}
    missing: list[str] = []
    for dataset_class in build_adapter_map(include_private=True):
        name = dataset_class.__name__
        root = osp.join(data_root, dataset_default_root(dataset_class))
        if not osp.isdir(osp.join(root, "processed")):
            missing.append(name)
            continue
        read_records = gaps.interned_entries(root) == 0 or (
            gaps.carries_a_compound_phenotype(dataset_class)
        )
        held: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "references": 0,
                "with_inchikey": 0,
                "inchikeys": set(),
                "without_cid": 0,
                "inchikey_gap_reasons": set(),
            }
        )
        compounds = gaps.store_compounds(root, read_records)
        while True:
            try:
                compound = next(compounds)
            except StopIteration:
                break
            except AttributeError as error:
                unreadable[name] = f"{type(error).__name__}: {error}"
                break
            label = str(compound["name"])
            fold = label.strip().lower()
            if fold not in keys:
                continue
            entry = held[label]
            entry["references"] += 1
            if compound["inchikey"] is not None:
                entry["with_inchikey"] += 1
                entry["inchikeys"].add(compound["inchikey"])
            if compound.get("pubchem_cid") is None:
                entry["without_cid"] += 1
            entry["inchikey_gap_reasons"].update(
                str(gap["reason"])
                for gap in compound.get("provenance_gaps") or []
                if gap["field"] == "inchikey"
            )
        if not held:
            continue
        labels: dict[str, Any] = {}
        stale = False
        for label, entry in sorted(held.items()):
            now = resolve_compound_identity(name=label)
            differs = (now.name is not None and now.name != label) or any(
                key != now.inchikey for key in entry["inchikeys"]
            )
            gap_closes = (
                now.inchikey is not None
                and entry["with_inchikey"] < (entry["references"])
            )
            # A mixture row gains a CID but no InChIKey, and turns a recoverable gap into
            # a terminal one; only a compound the RESOLVER built carries an inchikey gap
            # (a media mixture is a bare Compound), so the reason is compared only there.
            now_gap = [
                str(gap.reason)
                for gap in resolved_compound(label).provenance_gaps
                if gap.field == "inchikey"
            ]
            cid_fills = now.pubchem_cid is not None and entry["without_cid"] > 0
            reason_moves = bool(entry["inchikey_gap_reasons"]) and (
                entry["inchikey_gap_reasons"] != set(now_gap)
            )
            differs = differs or (
                cid_fills and bool(entry["inchikey_gap_reasons"] | entry["inchikeys"])
            )
            differs = differs or reason_moves
            stale = stale or differs or gap_closes
            labels[label] = {
                "references": entry["references"],
                "with_inchikey": entry["with_inchikey"],
                "table_name_now": now.name,
                "table_status_now": now.status.value,
                "table_inchikey_now": now.inchikey,
                "store_differs_from_table": differs or gap_closes,
            }
        stores[name] = {"root": root, "stale": stale, "labels": labels}
        print(
            f"  {'STALE' if stale else 'fresh'} {name}: {len(labels)} curated labels",
            flush=True,
        )

    stale_classes = sorted(n for n, s in stores.items() if s["stale"])
    result = {
        "tag": args.tag,
        "table_sha256": _TABLE_SHA256,
        "curated_labels": len(keys),
        "stores_holding_a_curated_label": len(stores),
        "stale_stores": stale_classes,
        "unreadable_stores": unreadable,
        "mapped_datasets_without_a_store": missing,
        "stores": stores,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, f"compound_curator_726_860_store_reach_{args.tag}.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    if args.classes_file is not None:
        args.classes_file.write_text("".join(f"{n}\n" for n in stale_classes))
    print(f"stale stores ({len(stale_classes)}): {stale_classes}")
    print(f"unreadable: {sorted(unreadable)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
