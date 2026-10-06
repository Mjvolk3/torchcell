# experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.wildenhain2015_inputs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs
"""Measure what the #504 input fix changes in the Wildenhain 2015 CGM store.

Three passes, no database connection:

1. RAW (the sha256-pinned ``1159580.csv.gz`` in the raw mirror): rows, SIDs and ORFs of the
   release; rows and SIDs per non-ORF strain label (``NA`` / ``NULL``); whether the
   ``NA/NNK1`` SIDs and CIDs overlap the ones released under ``orf=YKL171W``; the 33
   SGD-essential ORFs (intersection with
   ``$DATA_ROOT/data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json``); and
   the per-strain least-squares fit of ``z_score`` on ``normalized OD average`` (r^2,
   the normalized OD at which z = 0, slope), the measurement behind the N(1, IQR) reading.
2. NEW store (``--new-root``, a built ``env_chemgen_wildenhain2015`` tree): record counts
   by genotype kind, the essential strains' perturbation class, the YKL171W records, the
   wild-type records, the reference(s), the background, the drop ledger.
3. OLD store (``--old-root``, the pre-fix build): record count, and for every
   (systematic ORF, compound name) key present in BOTH stores, whether the stored z is
   bit-identical.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/wildenhain2015_inputs.json``
and ``wildenhain2015_inputs_labels.csv``.

Run from the repo root::

    # before landing: the scratch build vs the dev-tree (pre-fix) build
    python experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs.py \
        --new-root <scratch>/env_chemgen_wildenhain2015
    # after the slurm rebuild: the rebuilt dev tree, with the retired store as old
    python experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs.py \
        --dev-lmdb --old-root <graveyard>/env_chemgen_wildenhain2015
"""

import argparse
import csv
import gzip
import json
import os
import os.path as osp
import pickle
import statistics
from collections import Counter, defaultdict
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.data import verify_sha256
from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae import wildenhain2015 as w

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
SLUG = "data/torchcell/env_chemgen_wildenhain2015"


def raw_pass(data_root: str) -> dict[str, Any]:
    """Counts and the z ~ normalized-OD fit read straight from the pinned export."""
    path = w.raw_mirror_dir(data_root) / w.DATA_REL
    verify_sha256(path, w.DATA_SHA256)
    n_rows = 0
    sids: set[str] = set()
    orfs: set[str] = set()
    label_rows: Counter[str] = Counter()
    label_sids: dict[str, set[str]] = defaultdict(set)
    label_cids: dict[str, set[str]] = defaultdict(set)
    ykl171w_sids: set[str] = set()
    ykl171w_cids: set[str] = set()
    fit: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        next(reader)
        for row in reader:
            if not row:
                continue
            n_rows += 1
            orf = row[idx["orf"]].strip()
            sym = row[idx["sym"]].strip()
            sid = row[idx["PUBCHEM_SID"]].strip()
            cid = row[idx["PUBCHEM_CID"]].strip()
            sids.add(sid)
            if w._SYSTEMATIC_RE.match(orf):
                orfs.add(orf)
                strain = orf
            else:
                label = f"{orf}/{sym}"
                label_rows[label] += 1
                label_sids[label].add(sid)
                if cid:
                    label_cids[label].add(cid)
                strain = label
            if orf == "YKL171W":
                ykl171w_sids.add(sid)
                if cid:
                    ykl171w_cids.add(cid)
            z = row[idx["z_score"]].strip()
            od = row[idx["normalized OD average"]].strip()
            if z and od:
                fit[strain].append((float(od), float(z)))
    essential = set(
        json.loads(
            open(osp.join(data_root, w.ESSENTIAL_GENE_SET_SOURCE.source_uri)).read()
        )
    )
    fits = []
    for points in fit.values():
        if len(points) < 50:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if statistics.pvariance(xs) == 0:
            continue
        slope, intercept = statistics.linear_regression(xs, ys)
        r = statistics.correlation(xs, ys)
        fits.append((r * r, -intercept / slope, slope))
    nnk1 = "NA/NNK1"
    return {
        "data_rows": n_rows,
        "distinct_sids": len(sids),
        "systematic_orfs": len(orfs),
        "non_orf_label_rows": dict(sorted(label_rows.items())),
        "non_orf_label_sids": {k: len(v) for k, v in sorted(label_sids.items())},
        "nnk1_sids_shared_with_ykl171w": len(label_sids[nnk1] & ykl171w_sids),
        "nnk1_cids_shared_with_ykl171w": len(label_cids[nnk1] & ykl171w_cids),
        "ykl171w_orf_labelled_sids": len(ykl171w_sids),
        "essential_orfs": sorted(orfs & essential),
        "essential_orfs_match_loader_pin": orfs & essential
        == set(w.ESSENTIAL_GENE_ORFS),
        "z_vs_normalized_od_fit": {
            "n_strain_screens_fitted": len(fits),
            "rule": "per strain label with >= 50 rows: least-squares z = a + b * normOD",
            "median_r2": statistics.median(f[0] for f in fits),
            "median_normalized_od_at_z0": statistics.median(f[1] for f in fits),
            "mean_normalized_od_at_z0": statistics.fmean(f[1] for f in fits),
            "median_slope": statistics.median(f[2] for f in fits),
        },
    }


def read_store(root: str) -> list[dict[str, Any]]:
    """Every record of a built store, interned sub-objects spliced back in."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            for key, value in txn.cursor():
                interned[key.decode()] = pickle.loads(value)
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    records = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            records.append(resolve_interned(pickle.loads(value), interned))
    env.close()
    return records


def _key(record: dict[str, Any]) -> tuple[str, str]:
    perturbations = record["experiment"]["genotype"]["perturbations"]
    strain = perturbations[0]["systematic_gene_name"] if perturbations else "wild type"
    compound = record["experiment"]["environment"]["perturbations"][0]["compound"]
    return strain, compound["name"]


def new_pass(root: str) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    """What the fixed store holds."""
    records = read_store(root)
    kinds: Counter[str] = Counter()
    essential_types: Counter[str] = Counter()
    essential_gap_fields: Counter[tuple[str, ...]] = Counter()
    ykl171w = 0
    wild_type = 0
    references: dict[str, dict[str, Any]] = {}
    reference_ids: set[int] = set()
    solvent_percent: Counter[float] = Counter()
    experiment_types: Counter[str] = Counter()
    z: dict[tuple[str, str], float] = {}
    for record in records:
        experiment = record["experiment"]
        experiment_types[experiment["experiment_type"]] += 1
        perturbations = experiment["genotype"]["perturbations"]
        if not perturbations:
            kinds["wild_type (empty genotype)"] += 1
            wild_type += 1
        else:
            p = perturbations[0]
            kinds[p["perturbation_type"]] += 1
            if p["systematic_gene_name"] in w.ESSENTIAL_GENE_ORFS:
                essential_types[p["perturbation_type"]] += 1
                essential_gap_fields[
                    tuple(g["field"] for g in p.get("provenance_gaps", []))
                ] += 1
            if p["systematic_gene_name"] == "YKL171W":
                ykl171w += 1
        for perturbation in experiment["environment"]["perturbations"]:
            solvent_percent[perturbation["solvent"]["percent"]] += 1
        # interned references are shared objects, so serialize each distinct one once
        if id(record["reference"]) not in reference_ids:
            reference_ids.add(id(record["reference"]))
            references[json.dumps(record["reference"], sort_keys=True)] = record[
                "reference"
            ]
        z[_key(record)] = experiment["phenotype"]["environment_response"]
    reference = next(iter(references.values()))
    background = reference["genome_reference"]["background"]
    ledger = json.loads(
        open(osp.join(root, "preprocess", "dropped_records.json")).read()
    )
    return {
        "records": len(records),
        "experiment_types": dict(experiment_types),
        "records_by_genotype_kind": dict(kinds),
        "essential_strain_records_by_perturbation_type": dict(essential_types),
        "essential_strain_gap_fields": {
            ",".join(k): v for k, v in essential_gap_fields.items()
        },
        "essential_strains_served": sorted(
            {_key(r)[0] for r in records if _key(r)[0] in w.ESSENTIAL_GENE_ORFS}
        ),
        "ykl171w_records": ykl171w,
        "wild_type_records": wild_type,
        "distinct_references": len(references),
        "reference": {
            "experiment_reference_type": reference["experiment_reference_type"],
            "environment_perturbations": reference["environment_reference"][
                "perturbations"
            ],
            "environment_response": reference["phenotype_reference"][
                "environment_response"
            ],
            "units": reference["phenotype_reference"]["units"],
        },
        "background": {
            "name": background["name"],
            "mating_type": background["mating_type"],
            "ploidy": background["ploidy"],
            "alleles": [
                {
                    "allele_name": a["allele_name"],
                    "systematic_gene_name": a["systematic_gene_name"],
                    "edit": a["edit"],
                    "sourced_by": [
                        sv["provenance"]["citation_key"] for sv in a["provenance"]
                    ],
                    "gaps": [g["field"] for g in a["provenance_gaps"]],
                }
                for a in background["alleles"]
            ],
            "provenance_citation_keys": [
                sv["provenance"]["citation_key"] for sv in background["provenance"]
            ],
        },
        "solvent_percent_counts": {str(k): v for k, v in solvent_percent.items()},
        "drop_ledger": ledger,
    }, z


def old_pass(root: str, new_z: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Record count of the pre-fix store and z identity on the shared keys."""
    records = read_store(root)
    old_z = {
        _key(r): r["experiment"]["phenotype"]["environment_response"] for r in records
    }
    shared = set(old_z) & set(new_z)
    changed = sorted(k for k in shared if old_z[k] != new_z[k])
    return {
        "records": len(records),
        "keys": len(old_z),
        "keys_shared_with_new": len(shared),
        "keys_only_in_old": len(set(old_z) - set(new_z)),
        "keys_only_in_new": len(set(new_z) - set(old_z)),
        "only_in_new_by_strain": dict(
            Counter(k[0] for k in set(new_z) - set(old_z)).most_common(10)
        ),
        "shared_keys_with_a_different_z": len(changed),
        "changed_examples": [
            {"strain": k[0], "compound": k[1], "old": old_z[k], "new": new_z[k]}
            for k in changed[:20]
        ],
    }


def main() -> None:
    """Run the three passes and write the results."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", default=None)
    parser.add_argument("--old-root", default=osp.join(data_root, SLUG))
    parser.add_argument(
        "--dev-lmdb",
        action="store_true",
        help="read the rebuilt dev tree as the NEW store",
    )
    args = parser.parse_args()
    new_root = osp.join(data_root, SLUG) if args.dev_lmdb else args.new_root
    if new_root is None:
        parser.error("pass --new-root or --dev-lmdb")
    out: dict[str, Any] = {"raw": raw_pass(data_root), "new_root": new_root}
    out["new"], new_z = new_pass(new_root)
    if osp.abspath(args.old_root) != osp.abspath(new_root):
        out["old_root"] = args.old_root
        out["old"] = old_pass(args.old_root, new_z)
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "wildenhain2015_inputs.json"), "w") as handle:
        json.dump(out, handle, indent=2, default=str)
    with open(
        osp.join(RESULTS, "wildenhain2015_inputs_labels.csv"), "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "released_rows", "released_sids", "disposition"])
        for label, rows in out["raw"]["non_orf_label_rows"].items():
            orf, sym = label.split("/", 1)
            writer.writerow(
                [
                    label,
                    rows,
                    out["raw"]["non_orf_label_sids"][label],
                    w.NON_ORF_STRAIN_LABELS[(orf, sym)].disposition,
                ]
            )
    print(json.dumps(out, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
