# experiments/036-dataset-fixes-before-kg-build/scripts/synth_leth_db_entrez_resolution.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.synth_leth_db_entrez_resolution]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/synth_leth_db_entrez_resolution
"""Measure the SynLethDB Entrez-id gene resolution fix (#597).

Builds ``SynthLethalityYeastSynthLethDbDataset`` and ``SynthRescueYeastSynthLethDbDataset``
into a scratch root (``--build``; the raw CSVs are symlinked from the dev tree's
``raw/``, nothing in the dev tree is written), then reads the scratch stores and the
dev-tree stores ``$DATA_ROOT/data/torchcell/synth_{lethality,rescue}_yeast_synth_leth_db``
read-only and reports, per dataset:

- source rows, kept records, and the drop ledger (``preprocess/dropped_records.json``);
- old-store rows whose stored ORF pair disagrees with the Entrez-id ORF pair (every side
  whose Entrez id the pinned NCBI GFF maps), which is the issue's 872 / 119 count;
- of the kept rows, how many changed ORF pair between the dev store and the new store,
  and whether every change is one of the old disagreements;
- duplicate unordered ORF pairs in each store;
- the ORFs each of the issue's ten names (and ``IMP2'``) is stored under, old and new;
- which genome file the name cross-check read (``data.db``) and its sha256 and content
  digest, beside ``data.db.bak`` and ``data_alt.db``, plus the NCBI GFF sha256.

Old-store LMDB keys are source row indices (the old loader wrote every row); new-store
keys are contiguous over kept rows, mapped back through the ledger's dropped rows.

Writes ``results/synth_leth_db_entrez_resolution.json`` and
``results/synth_leth_db_entrez_resolution_changed_rows.csv``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/synth_leth_db_entrez_resolution.py \
        --scratch-root <dir> [--build]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from torchcell.data import file_sha256
from torchcell.datamodels.schema import SgaKanMxDeletionPerturbation
from torchcell.datasets.scerevisiae import synth_leth_db as s
from torchcell.sequence.genome.scerevisiae.s288c import (
    SCerevisiaeGenome,
    database_content_digest,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
SLUGS = {
    "SL": ("synth_lethality_yeast_synth_leth_db", s.SL_CSV_NAME),
    "SR": ("synth_rescue_yeast_synth_leth_db", s.SR_CSV_NAME),
}
#: The issue's table (name -> correct ORF), plus the primed IMP2'.
ISSUE_NAMES = {
    "STM1": "YLR150W",
    "SDC1": "YDR469W",
    "MFT1": "YML062C",
    "NSP1": "YJL041W",
    "RPL37A": "YLR185W",
    "CCS1": "YMR038C",
    "YPK1": "YKL126W",
    "TAF1": "YGR274C",
    "SSL2": "YIL143C",
    "HAP1": "YLR256W",
    "IMP2'": "YIL154C",
}


def _records(lmdb_dir: str) -> dict[int, dict[str, Any]]:
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    out: dict[int, dict[str, Any]] = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            out[int(key.decode())] = pickle.loads(value)
    env.close()
    return out


def _sides(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (p["systematic_gene_name"], p["perturbed_gene_name"])
        for p in record["experiment"]["genotype"]["perturbations"]
    ]


def _pair(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(orf for orf, _ in _sides(record)))


def _build(scratch_root: str, data_root: str, genome: SCerevisiaeGenome) -> None:
    for label, (slug, csv_name) in SLUGS.items():
        root = osp.join(scratch_root, slug)
        raw = osp.join(root, "raw")
        os.makedirs(raw, exist_ok=True)
        link = osp.join(raw, csv_name)
        if not osp.lexists(link):
            os.symlink(
                osp.join(data_root, "data/torchcell", slug, "raw", csv_name), link
            )
        cls = (
            s.SynthLethalityYeastSynthLethDbDataset
            if label == "SL"
            else s.SynthRescueYeastSynthLethDbDataset
        )
        print(cls(root=root, genome=genome))


def _measure(
    label: str, scratch_root: str, data_root: str, entrez_to_orf: dict[int, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    slug, csv_name = SLUGS[label]
    dev = osp.join(data_root, "data/torchcell", slug)
    new = osp.join(scratch_root, slug)
    raw = pd.read_csv(
        osp.join(dev, "raw", csv_name),
        dtype={"r.pubmed_id": str, "n1.identifier": "int64", "n2.identifier": "int64"},
    )
    old = _records(osp.join(dev, "processed/lmdb"))
    fresh = _records(osp.join(new, "processed/lmdb"))
    with open(osp.join(new, "preprocess/dropped_records.json")) as f:
        ledger = s.DropLog.model_validate_json(f.read())
    dropped = {r.source_row: rule.rule for rule in ledger.rules for r in rule.rows}
    kept_rows = [i for i in range(len(raw)) if i not in dropped]
    assert len(kept_rows) == len(fresh) == ledger.kept_records
    new_by_row = {row: fresh[key] for key, row in enumerate(kept_rows)}

    # Old-store rows whose stored ORF pair disagrees with the Entrez-id ORF pair.
    old_wrong: set[int] = set()
    for i, (e1, e2) in enumerate(zip(raw["n1.identifier"], raw["n2.identifier"])):
        stored = Counter(orf for orf, _ in _sides(old[i]))
        expected = [entrez_to_orf.get(int(e)) for e in (e1, e2)]
        known = Counter(o for o in expected if o is not None)
        if any(stored[o] < n for o, n in known.items()):
            old_wrong.add(i)

    changed: list[dict[str, Any]] = []
    records = raw.to_dict("records")
    for row in kept_rows:
        if _pair(old[row]) != _pair(new_by_row[row]):
            rec = records[row]
            changed.append(
                {
                    "source_row": row,
                    "n1.name": rec["n1.name"],
                    "n1.identifier": int(rec["n1.identifier"]),
                    "n2.name": rec["n2.name"],
                    "n2.identifier": int(rec["n2.identifier"]),
                    "old_pair": ";".join(_pair(old[row])),
                    "new_pair": ";".join(_pair(new_by_row[row])),
                    "was_old_disagreement": row in old_wrong,
                }
            )
    changed_rows = {c["source_row"] for c in changed}

    def duplicates(records: list[dict[str, Any]]) -> int:
        pairs = Counter(frozenset(_pair(r)) for r in records)
        return sum(n - 1 for n in pairs.values() if n > 1)

    names: dict[str, dict[str, list[str]]] = {}
    for name in ISSUE_NAMES:
        spelled = SgaKanMxDeletionPerturbation.model_validate(
            {
                "systematic_gene_name": "YAL001C",
                "perturbed_gene_name": name,
                "strain_id": "S288C",
            }
        ).perturbed_gene_name
        names[name] = {
            "old": sorted(
                {o for r in old.values() for o, n in _sides(r) if n == spelled}
            ),
            "new": sorted(
                {o for r in fresh.values() for o, n in _sides(r) if n == spelled}
            ),
        }

    summary = {
        "source_rows": len(raw),
        "old_records": len(old),
        "new_records": len(fresh),
        "ledger": {
            rule.rule: {
                "n_records": rule.n_records,
                "source_rows": [r.source_row for r in rule.rows],
            }
            for rule in ledger.rules
        },
        "old_rows_disagreeing_with_entrez": len(old_wrong),
        "kept_rows_changed_orf_pair": len(changed),
        "changed_rows_all_old_disagreements": changed_rows <= old_wrong,
        "old_disagreements_not_changed_and_kept": sorted(
            old_wrong - changed_rows - set(dropped)
        ),
        "old_disagreements_dropped": {
            str(row): dropped[row] for row in sorted(old_wrong & set(dropped))
        },
        "duplicate_orf_pairs_old": duplicates(list(old.values())),
        "duplicate_orf_pairs_new": duplicates(list(fresh.values())),
        "issue_names": names,
    }
    return summary, changed


def main() -> None:
    """Optionally build the scratch stores, then measure both against the dev tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    data_root = os.environ["DATA_ROOT"]
    genome_root = osp.join(data_root, "data/sgd/genome")
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=osp.join(data_root, "data/go"), overwrite=False
    )
    if args.build:
        _build(args.scratch_root, data_root, genome)
    gff = s.ncbi_gff_path(genome)
    entrez_to_orf = s.load_entrez_to_orf(gff)
    genome_files = {
        name: {
            "sha256": file_sha256(osp.join(genome_root, name)),
            "content_digest": database_content_digest(osp.join(genome_root, name)),
            "n_features_chrmt": _chrmt(osp.join(genome_root, name)),
        }
        for name in ("data.db", "data.db.bak", "data_alt.db")
    }
    out: dict[str, Any] = {
        "ncbi_gff": {
            "path": gff,
            "sha256": file_sha256(gff),
            "n_entrez": len(entrez_to_orf),
        },
        "genome_db_read": osp.join(genome_root, "data.db"),
        "genome_files": genome_files,
    }
    rows: list[dict[str, Any]] = []
    for label in SLUGS:
        summary, changed = _measure(label, args.scratch_root, data_root, entrez_to_orf)
        out[label] = summary
        rows.extend({"dataset": label, **c} for c in changed)
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "synth_leth_db_entrez_resolution.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS, "synth_leth_db_entrez_resolution_changed_rows.csv"),
        index=False,
    )
    print(json.dumps(out, indent=2))


def _chrmt(db_path: str) -> int:
    """Count ``gene`` features on chrmt in one gffutils database (read-only)."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    (n,) = conn.execute(
        "SELECT COUNT(*) FROM features WHERE featuretype = 'gene' AND seqid = 'chrmt'"
    ).fetchone()
    conn.close()
    return int(n)


if __name__ == "__main__":
    main()
