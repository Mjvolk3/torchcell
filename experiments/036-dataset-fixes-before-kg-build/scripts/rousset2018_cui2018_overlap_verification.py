# experiments/036-dataset-fixes-before-kg-build/scripts/rousset2018_cui2018_overlap_verification.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.rousset2018_cui2018_overlap_verification]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/rousset2018_cui2018_overlap_verification
"""Verify from the BUILT dev stores that Rousset 2018 and Cui 2018 store nothing twice (#760).

#760 measured, from the two pinned raw tables, that Rousset 2018's growth screen IS Cui
2018's ``fit75`` screen (Pearson r 1.0000 over 54,326 shared 20-nt spacers) and asked for
one of the two releases to stop storing it. The Rousset loader's answer landed with the
loader: it serves only the four phage-derived screens and drops every S1 Table row under a
named rule. This script proves that from the stores rather than from the docstring:

1. **Per-screen record counts** of both dev LMDBs, keyed on
   ``experiment.phenotype.screen_id``. Rousset must carry only the four phage screens and
   no growth screen; Cui must carry both dose arms.
2. **Experiment content ids.** The id a knowledge-graph build writes is
   ``sha256(json.dumps(experiment.model_dump()))``
   (``torchcell/adapters/cell_adapter.py::_experiment_node``). The two stores' id sets must
   be disjoint, and each store's ids must be as numerous as its records.
3. **(spacer, screen_id) keys.** The pair that identifies one released measurement. The two
   stores must share none, and no spacer may carry a growth-screen record in Rousset.
4. **Spacer-level overlap.** How many of Rousset's stored spacers Cui's store also carries
   (expected: many, since the phage library is a subset of the same guide library) and how
   many (spacer, screen_id) measurements that amounts to (expected: 0 shared).
5. **The retention ledger's arithmetic**, re-added from
   ``preprocess/dropped_records.json``: the four S1 rules must sum to the 59,246 rows S1
   released, and kept + dropped must equal the 128,126 released (guide, screen) cells.
6. **The headline join, re-measured from the pinned raw bytes**: Rousset S1 ``log2FC``
   against Cui ``fit75`` and ``fit18``, joined on the 20-nt spacer, with each file's sha256
   verified against the loader's pin first.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/rousset2018_cui2018_overlap_verification.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/rousset2018_cui2018_overlap_verification.py
"""

import hashlib
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.ecoli.cui2018 import SCREEN_FILENAME as CUI_FILENAME
from torchcell.datasets.ecoli.cui2018 import SCREEN_SHA256 as CUI_SHA256
from torchcell.datasets.ecoli.cui2018 import CrispriKnockdownCui2018Dataset
from torchcell.datasets.ecoli.rousset2018 import TABLE_SHA256 as ROUSSET_TABLE_SHA256
from torchcell.datasets.ecoli.rousset2018 import CrispriScreenRousset2018Dataset

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
ROUSSET_SLUG = "ecoli_crispri_rousset2018"
CUI_SLUG = "crispri_knockdown_cui2018"
#: Rousset's S1 Table, the growth screen (``pgen.1007749.s011.csv``).
ROUSSET_GROWTH_FILENAME = "pgen.1007749.s011.csv"
#: The four screen tags the Rousset store is allowed to carry.
ROUSSET_PHAGE_SCREENS = frozenset(
    {"phage_lambda", "phage_T4", "phage_186cIts", "lambda_transduction"}
)
#: The two Cui dose arms.
CUI_SCREENS = frozenset({"LC-E18", "LC-E75"})
#: S1 Table rows released, and (guide, screen) cells released, from the loader's ledger.
S1_RELEASED_ROWS = 59246
RELEASED_CELLS = 128126


def _sha256(path: str) -> str:
    """Hex sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def store_fingerprint(dataset: Any) -> dict[str, Any]:
    """Per-screen counts, content ids and (spacer, screen) keys of one built store."""
    experiment_class = dataset.experiment_class
    screens: Counter[str] = Counter()
    content_ids: set[str] = set()
    keys: set[tuple[str, str]] = set()
    spacers: set[str] = set()
    for index in range(len(dataset)):
        raw = dataset[index]["experiment"]
        dump = (
            raw.model_dump()
            if hasattr(raw, "model_dump")
            else experiment_class(**raw).model_dump()
        )
        content_ids.add(hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest())
        screen = dump["phenotype"]["screen_id"]
        screens[screen] += 1
        # One record is one guide. Cui writes a perturbation per gene a multi-target
        # guide hits, so a record can carry several perturbations, all on ONE spacer.
        guides = {
            p["crispr"]["guide_sequence"] for p in dump["genotype"]["perturbations"]
        }
        assert len(guides) == 1, f"record {index} carries {len(guides)} spacers"
        spacer = guides.pop()
        spacers.add(spacer)
        keys.add((spacer, screen))
    return {
        "records": len(dataset),
        "screen_id": dict(sorted(screens.items())),
        "content_ids": content_ids,
        "spacer_screen_keys": keys,
        "spacers": spacers,
    }


def released_join(rousset_path: str, cui_path: str) -> dict[str, Any]:
    """Re-measure the #760 headline join on the pinned raw bytes."""
    s1 = pd.read_csv(rousset_path)
    cui = pd.read_csv(cui_path)
    s1 = s1[["target", "log2FC"]].dropna()
    cui = cui[["guide", "fit18", "fit75"]].dropna()
    # One row per spacer: Cui releases a guide once per perfect chromosomal match with
    # fit18/fit75 constant across those rows (cui2018.py), so first() is that value.
    cui = cui.groupby("guide", as_index=False).first()
    merged = s1.merge(cui, left_on="target", right_on="guide", how="inner")
    out: dict[str, Any] = {
        "rousset_s1_spacers_with_a_value": int(s1["target"].nunique()),
        "cui_spacers_numeric_in_both_columns": int(cui["guide"].nunique()),
        "shared_spacers": int(len(merged)),
        "rousset_only": int(s1["target"].nunique() - len(merged)),
        "cui_only": int(cui["guide"].nunique() - len(merged)),
    }
    for column in ("fit75", "fit18"):
        difference = (merged["log2FC"] - merged[column]).abs()
        out[f"vs_{column}"] = {
            "n": int(len(merged)),
            "median_abs_diff": round(float(difference.median()), 4),
            "max_abs_diff": round(float(difference.max()), 4),
            "pearson_r": round(
                float(np.corrcoef(merged["log2FC"], merged[column])[0, 1]), 4
            ),
        }
    return out


def main() -> None:
    """Measure both stores and the released join, then write the results JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    rousset_root = osp.join(data_root, "data/torchcell", ROUSSET_SLUG)
    cui_root = osp.join(data_root, "data/torchcell", CUI_SLUG)

    rousset_dataset = CrispriScreenRousset2018Dataset(root=rousset_root)
    rousset = store_fingerprint(rousset_dataset)
    rousset_dataset.close_lmdb()
    cui_dataset = CrispriKnockdownCui2018Dataset(root=cui_root)
    cui = store_fingerprint(cui_dataset)
    cui_dataset.close_lmdb()

    ledger = json.loads(
        open(osp.join(rousset_root, "preprocess", "dropped_records.json")).read()
    )
    by_rule = {rule["rule"]: rule["n_records"] for rule in ledger["rules"]}
    s1_rules = [
        "guide_targets_no_gene",
        "guide_targets_the_template_strand",
        "growth_screen_measurement_is_served_by_cui2018",
        "growth_screen_guide_is_below_cui2018_read_floor",
    ]

    rousset_raw = osp.join(rousset_root, "raw")
    raw_digests = {
        name: _sha256(osp.join(rousset_raw, name)) for name in ROUSSET_TABLE_SHA256
    }
    cui_raw_path = osp.join(cui_root, "raw", CUI_FILENAME)

    result: dict[str, Any] = {
        "dev_stores": {
            "CrispriScreenRousset2018Dataset": {
                "root": rousset_root,
                "records": rousset["records"],
                "screen_id": rousset["screen_id"],
                "distinct_content_ids": len(rousset["content_ids"]),
                "distinct_spacers": len(rousset["spacers"]),
            },
            "CrispriKnockdownCui2018Dataset": {
                "root": cui_root,
                "records": cui["records"],
                "screen_id": cui["screen_id"],
                "distinct_content_ids": len(cui["content_ids"]),
                "distinct_spacers": len(cui["spacers"]),
            },
        },
        "overlap": {
            "shared_content_ids": len(rousset["content_ids"] & cui["content_ids"]),
            "shared_spacer_screen_keys": len(
                rousset["spacer_screen_keys"] & cui["spacer_screen_keys"]
            ),
            "shared_spacers": len(rousset["spacers"] & cui["spacers"]),
            "rousset_screens_outside_the_four_phage_screens": sorted(
                set(rousset["screen_id"]) - ROUSSET_PHAGE_SCREENS
            ),
            "cui_screens_outside_the_two_dose_arms": sorted(
                set(cui["screen_id"]) - CUI_SCREENS
            ),
        },
        "retention_ledger": {
            "source_records": ledger["source_records"],
            "kept_records": ledger["kept_records"],
            "dropped_records": ledger["dropped_records"],
            "kept_plus_dropped": ledger["kept_records"] + ledger["dropped_records"],
            "released_cells": RELEASED_CELLS,
            "s1_rule_counts": {rule: by_rule[rule] for rule in s1_rules},
            "s1_rules_sum": sum(by_rule[rule] for rule in s1_rules),
            "s1_released_rows": S1_RELEASED_ROWS,
            "growth_screen_rule_served_by": by_rule
            and next(
                rule["served_by"]
                for rule in ledger["rules"]
                if rule["rule"] == "growth_screen_measurement_is_served_by_cui2018"
            ),
        },
        "raw_sha256": {
            "rousset_pinned_match": {
                name: raw_digests[name] == ROUSSET_TABLE_SHA256[name]
                for name in sorted(ROUSSET_TABLE_SHA256)
            },
            "cui_pinned_match": _sha256(cui_raw_path) == CUI_SHA256,
        },
        "released_join": released_join(
            osp.join(rousset_raw, ROUSSET_GROWTH_FILENAME), cui_raw_path
        ),
    }
    result["verdict"] = {
        "rousset_stores_no_growth_screen": not result["overlap"][
            "rousset_screens_outside_the_four_phage_screens"
        ],
        "zero_content_id_overlap": result["overlap"]["shared_content_ids"] == 0,
        "zero_spacer_screen_overlap": result["overlap"]["shared_spacer_screen_keys"]
        == 0,
        "s1_rules_account_for_every_released_row": result["retention_ledger"][
            "s1_rules_sum"
        ]
        == S1_RELEASED_ROWS,
        "kept_plus_dropped_equals_released_cells": result["retention_ledger"][
            "kept_plus_dropped"
        ]
        == RELEASED_CELLS,
        "every_raw_file_matches_its_pin": all(
            result["raw_sha256"]["rousset_pinned_match"].values()
        )
        and result["raw_sha256"]["cui_pinned_match"],
    }

    os.makedirs(RESULTS, exist_ok=True)
    out_path = osp.join(RESULTS, "rousset2018_cui2018_overlap_verification.json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
