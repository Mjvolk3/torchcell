# experiments/036-dataset-fixes-before-kg-build/scripts/borchert2024_rch2_records.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.borchert2024_rch2_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/borchert2024_rch2_records
"""Count the rebuilt Borchert 2024 store's records by medium and by source publication (#860).

#860 expects the store to grow from 1,372,280 records to 1,372,280 + 94,640 once the
RCH2 medium is in the library, and the ten Rand 2017 samples to reach records carrying Rand
2017's publication (10 x 4,732 = 47,320). This reads every record of the dev store's LMDB
(read-only) and counts, per medium name and per publication DOI, how many there are and
how many distinct screens (samples) carry them, so both claims are measured from the
store rather than from the loader's own ledger. A record's environment is interned (a
``{"$ref": <sha256>}`` pointer into ``processed/interned``), so each distinct pointer is
resolved once against that env and its medium name read from the stored object.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/borchert2024_rch2_records.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/borchert2024_rch2_records.py
"""

import json
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict

import lmdb
from dotenv import load_dotenv

from torchcell.datamodels.media import RCH2_DEFINED_NOCARBON_PRICE2018
from torchcell.datasets.pputida.borchert2024 import SOURCE_STUDIES

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
STORE = "data/torchcell/rbtnseq_borchert2024"


def main() -> None:
    """Stream the LMDB once and write the counts."""
    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], STORE)
    env = lmdb.open(
        osp.join(root, "processed", "lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    interned = lmdb.open(
        osp.join(root, "processed", "interned"), readonly=True, lock=False
    )
    medium_of_ref: dict[str, str] = {}
    by_medium: Counter[str] = Counter()
    by_doi: Counter[str] = Counter()
    screens_by_doi: dict[str, set[str]] = defaultdict(set)
    rch2_screens: set[str] = set()
    with env.begin() as txn:
        entries = int(txn.stat()["entries"])
        for _key, value in txn.cursor():
            record = pickle.loads(value)
            experiment = record["experiment"]
            ref = str(experiment["environment"]["$ref"])
            if ref not in medium_of_ref:
                with interned.begin() as itxn:
                    stored = pickle.loads(itxn.get(ref.encode()))
                medium_of_ref[ref] = str(stored["media"]["name"])
            medium = medium_of_ref[ref]
            doi = str(record["publication"]["doi"])
            screen = str(experiment["phenotype"]["screen_id"])
            by_medium[medium] += 1
            by_doi[doi] += 1
            screens_by_doi[doi].add(screen)
            if medium == RCH2_DEFINED_NOCARBON_PRICE2018.name:
                rch2_screens.add(screen)
    env.close()
    interned.close()

    rand_doi = SOURCE_STUDIES["rand2017"].publication.doi
    result = {
        "store": STORE,
        "lmdb_entries": entries,
        "records_by_medium": dict(by_medium.most_common()),
        "records_by_publication_doi": dict(by_doi.most_common()),
        "screens_by_publication_doi": {
            doi: len(screens) for doi, screens in sorted(screens_by_doi.items())
        },
        "rch2_screens": sorted(rch2_screens),
        "rch2_records": by_medium[RCH2_DEFINED_NOCARBON_PRICE2018.name],
        "rand2017_doi": rand_doi,
        "rand2017_records": by_doi[str(rand_doi)],
        "rand2017_screens": sorted(screens_by_doi[str(rand_doi)]),
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "borchert2024_rch2_records.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "rch2_screens"}, indent=2)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
