# experiments/031-env-chemgen-inhibitor-tolerance/scripts/approved_drug_coverage.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.approved_drug_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/approved_drug_coverage
"""How many dosed compounds of the four kept datasets are approved medicines.

"Approved medicine" is read as ChEMBL ``max_phase == 4``: the molecule is approved for
therapeutic use in at least one major jurisdiction (ChEMBL takes this from the FDA Orange
Book, EMA, the Japanese and other national registries, including over-the-counter drugs).
That is the closest structured answer to "something a patient could buy". It is not a
count of drugs sold today, because ``max_phase`` stays 4 after a withdrawal, so the
``withdrawn_flag`` is reported beside it.

Each dosed InChIKey from the flattened served records (``results/records_<dataset>.parquet``)
is looked up in ChEMBL by exact standard InChIKey, 50 keys per request. An exact key match
misses a compound dosed as a salt or stereoisomer that ChEMBL registers under a different
key; that is an undercount, reported as such, not corrected.

The API responses are written verbatim to ``results/chembl_lookup.json`` with the request
URL base and retrieval time, so the counts can be recomputed without the network and an
upstream change can be detected. Writes ``results/approved_drug_coverage.csv`` (one row per
dataset and the union) and ``results/approved_drugs.csv`` (one row per approved compound).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time
from datetime import UTC, datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
FIELDS = (
    "molecule_chembl_id,pref_name,max_phase,withdrawn_flag,first_approval,"
    "molecule_structures"
)
#: The four datasets kept by the 031 planning document; Hillenmeyer HOM is dropped.
DATASETS: dict[str, str] = {
    "vanacloig2022": "Vanacloig 2022",
    "hillenmeyer2008_het": "Hillenmeyer HET",
    "hoepfner2014": "Hoepfner 2014",
    "wildenhain2015": "Wildenhain 2015",
}
BATCH = 50


def dataset_keys() -> dict[str, set[str]]:
    """Distinct dosed InChIKeys per dataset, from the flattened served records."""
    out: dict[str, set[str]] = {}
    for name in DATASETS:
        df = pd.read_parquet(
            osp.join(RESULTS_DIR, f"records_{name}.parquet"), columns=["inchikey"]
        ).drop_duplicates()
        keys: set[str] = set()
        for field in df["inchikey"].dropna():
            keys.update(k for k in str(field).split("|") if k and k != "None")
        out[name] = keys
    return out


def chembl_lookup(keys: list[str]) -> dict[str, list[dict]]:
    """InChIKey -> every ChEMBL molecule registered under it (usually zero or one)."""
    found: dict[str, list[dict]] = {k: [] for k in keys}
    for i in range(0, len(keys), BATCH):
        batch = keys[i : i + BATCH]
        resp = requests.get(
            CHEMBL,
            params={
                "molecule_structures__standard_inchi_key__in": ",".join(batch),
                "only": FIELDS,
                "limit": 1000,
            },
            timeout=120,
        )
        resp.raise_for_status()
        for mol in resp.json()["molecules"]:
            key = mol["molecule_structures"]["standard_inchi_key"]
            mol.pop("molecule_structures")
            found[key].append(mol)
        print(f"  {min(i + BATCH, len(keys))} of {len(keys)} keys", flush=True)
        time.sleep(0.2)
    return found


def is_approved(mols: list[dict]) -> bool:
    return any(
        m["max_phase"] is not None and float(m["max_phase"]) == 4.0 for m in mols
    )


def main() -> None:
    per_dataset = dataset_keys()
    union = sorted(set().union(*per_dataset.values()))
    print(f"{len(union)} distinct dosed InChIKeys across the four datasets")
    cache = osp.join(RESULTS_DIR, "chembl_lookup.json")
    found = chembl_lookup(union)
    with open(cache, "w") as f:
        json.dump(
            {
                "source": CHEMBL,
                "fields": FIELDS,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "molecules_by_inchikey": found,
            },
            f,
            indent=1,
        )

    approved = {k for k in union if is_approved(found[k])}
    withdrawn = {k for k in approved if any(m["withdrawn_flag"] for m in found[k])}
    rows = []
    for name, keys in [*per_dataset.items(), ("union", set(union))]:
        rows.append(
            {
                "dataset": DATASETS.get(name, "union of the four"),
                "compounds": len(keys),
                "in_chembl": sum(1 for k in keys if found[k]),
                "approved": len(keys & approved),
                "approved_withdrawn": len(keys & withdrawn),
                "approved_share": round(len(keys & approved) / len(keys), 3),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(osp.join(RESULTS_DIR, "approved_drug_coverage.csv"), index=False)

    drugs = []
    for k in sorted(approved):
        m = next(m for m in found[k] if float(m["max_phase"]) == 4.0)
        drugs.append(
            {
                "inchikey": k,
                "chembl_id": m["molecule_chembl_id"],
                "name": m["pref_name"],
                "first_approval": m["first_approval"],
                "withdrawn": bool(m["withdrawn_flag"]),
                "datasets": ";".join(
                    DATASETS[d] for d, ks in per_dataset.items() if k in ks
                ),
            }
        )
    pd.DataFrame(drugs).to_csv(osp.join(RESULTS_DIR, "approved_drugs.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
