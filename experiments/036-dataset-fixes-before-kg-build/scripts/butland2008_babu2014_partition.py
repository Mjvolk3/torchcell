# experiments/036-dataset-fixes-before-kg-build/scripts/butland2008_babu2014_partition.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.butland2008_babu2014_partition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/butland2008_babu2014_partition
"""Re-measure the Butland 2008 / Babu 2014 partition after Babu admitted its hypomorphs.

PR #837 moved Babu 2014's 3,420 hypomorph pairs from a drop rule onto the new
``BacterialMarkedAllelePerturbation`` leaf, so the served Babu store grew from 38,579 to
41,988 records. 398 of the admitted pairs carry ``screen_id="Butland et al."``, which is
the set Butland 2008's loader partitions itself against, so the served-by-Babu side of
that partition grew from 727 pairs to 1,125 and the Butland build stopped on its own pin
(slurm array 3565 task 53).

This script is the measurement behind the new pins. Nothing here is read from the
loader's docstring or from a build log: the served side comes from the rebuilt Babu dev
LMDB and the release side from the sha256-pinned Supplementary Table 4 workbook, and the
retention rules are re-applied in the loader's own order over the released cells.

What it measures:

1. **The served side.** Every record of the Babu dev store as an oriented
   (cat-marked donor, kan-marked recipient) gene pair, split by ``screen_id`` and by the
   perturbation leaf each side carries, so the 727 -> 1,125 growth is attributed to the
   marked-allele leaf rather than assumed from the PR body. The reciprocal unordered
   pairs are counted here too, because the Butland docstring cites that number for why
   the partition's unit is the ORIENTED pair.
2. **Where each served Butland-screen pair sits in the release.** Each of the 1,125 is
   located in the 39 x 8,073 matrix and classified by the FIRST Butland retention rule
   that applies to it, which is what decides whether it reaches the served rule at all.
3. **The new Butland ledger.** The six retention rules re-applied in order, giving the
   kept-record and kept-pair counts, and the cells the served rule now removes.
4. **No measurement stored twice.** The experiment content ids of both built stores (the
   id a knowledge-graph build writes,
   ``sha256(json.dumps(experiment.model_dump()))``, from
   ``torchcell/adapters/cell_adapter.py::_experiment_node``) must be disjoint, and so must
   the two stores' oriented gene pairs. This step needs the Butland store built, so it is
   skipped with a stated reason when it is absent.

Writes
``experiments/036-dataset-fixes-before-kg-build/results/butland2008_babu2014_partition.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/butland2008_babu2014_partition.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.data import verify_raw_files
from torchcell.datasets.bacteria_common import bacterial_genome
from torchcell.datasets.ecoli import butland2008 as b
from torchcell.datasets.ecoli.butland2008 import (
    BABU_ROOT_REL,
    BABU_SCREEN_TAG,
    DATASET_ROOT_REL,
    LABEL_NON_ESSENTIAL,
    QUERY_CASSETTE,
    RECIPIENT_CASSETTE,
    TABLE_S4,
)
from torchcell.sequence.genome.base import GeneNameStatus

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"


def served_side(served_root: str) -> dict[str, Any]:
    """Every served Babu record as an oriented pair, by screen and by leaf class."""
    from torchcell.verification.runners import stream_records

    pairs: dict[tuple[str, str], str] = {}
    screens: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    butland_leaves: Counter[str] = Counter()
    content_ids: set[str] = set()
    for record in stream_records(served_root):
        experiment = record["experiment"]
        by_cassette = {
            str(leaf["cassette"]): leaf
            for leaf in experiment["genotype"]["perturbations"]
        }
        pair = (
            str(by_cassette[QUERY_CASSETTE]["systematic_gene_name"]),
            str(by_cassette[RECIPIENT_CASSETTE]["systematic_gene_name"]),
        )
        if pair in pairs:
            raise RuntimeError(f"the served store holds {pair} twice")
        screen = str(experiment["phenotype"]["screen_id"])
        pairs[pair] = screen
        screens[screen] += 1
        signature = "+".join(
            f"{cassette}:{leaf['perturbation_type']}"
            for cassette, leaf in sorted(by_cassette.items())
        )
        leaves[signature] += 1
        if screen == BABU_SCREEN_TAG:
            butland_leaves[signature] += 1
        content_ids.add(
            hashlib.sha256(json.dumps(experiment).encode("utf-8")).hexdigest()
        )
    reciprocal = {frozenset(pair) for pair in pairs if (pair[1], pair[0]) in pairs}
    return {
        "served_root": served_root,
        "records": sum(screens.values()),
        "oriented_pairs": len(pairs),
        "reciprocal_unordered_pairs": len(reciprocal),
        "by_screen_id": dict(sorted(screens.items())),
        "by_perturbation_leaf": dict(sorted(leaves.items())),
        "butland_screen_by_perturbation_leaf": dict(sorted(butland_leaves.items())),
        "pairs": pairs,
        "content_ids": content_ids,
    }


def release_side(raw_dir: str) -> dict[str, Any]:
    """Parse the pinned Supplementary Table 4 workbook and the identifier rule sets."""
    verify_raw_files(raw_dir, b.DATA_SHA256)
    matrix_path = osp.join(raw_dir, TABLE_S4)
    scores = b.read_s_scores(matrix_path)
    raw_sizes = b.read_matrix_sheet(matrix_path, b.SHEET_RAW)
    order = [raw_sizes.query_tags.index(tag) for tag in scores.query_tags]
    colonies, zero_colonies, _ = b.colony_counts(raw_sizes)
    colonies, zero_colonies = colonies[:, order], zero_colonies[:, order]
    genome = bacterial_genome("ecoli", b.REFERENCE_STRAIN_NAME)
    names = sorted(set(scores.query_tags) | set(scores.recipient_tags))
    resolutions = {name: genome.resolve_gene_name(name) for name in names}
    not_a_tag = {
        name
        for name, resolved in resolutions.items()
        if resolved.status in (GeneNameStatus.RETIRED, GeneNameStatus.AMBIGUOUS)
    }
    remapped = {
        name: str(resolved.systematic_name)
        for name, resolved in resolutions.items()
        if name not in not_a_tag
        and resolved.systematic_name is not None
        and resolved.systematic_name != name
    }
    return {
        "scores": scores,
        "block": b.score_block(scores),
        "colonies": colonies,
        "zero_colonies": zero_colonies,
        "not_a_tag": not_a_tag,
        "remapped": remapped,
    }


def ledger(
    release: dict[str, Any], served: dict[tuple[str, str], str]
) -> dict[str, Any]:
    """Re-apply the six retention rules in the loader's order over every released cell.

    The per-cell rule the matrix assigns is also returned per oriented pair, which is
    what locates each served pair in the release.
    """
    scores = release["scores"]
    block, colonies, zero_colonies = (
        release["block"],
        release["colonies"],
        release["zero_colonies"],
    )
    not_a_tag, remapped = release["not_a_tag"], release["remapped"]
    rules: Counter[str] = Counter()
    pair_rule: dict[tuple[str, str], set[str]] = {}
    stored_pairs: set[tuple[str, str]] = set()
    kept = 0
    for row, recipient_tag in enumerate(scores.recipient_tags):
        label = scores.labels[row]
        for column, query_tag in enumerate(scores.query_tags):
            pair = (query_tag, recipient_tag)
            if label != LABEL_NON_ESSENTIAL:
                rule = b.RULE_SPA_TAG
            elif recipient_tag in not_a_tag:
                rule = b.RULE_NOT_A_TAG
            elif recipient_tag in remapped:
                rule = b.RULE_REMAPPED
            elif recipient_tag == query_tag:
                rule = b.RULE_SELF_PAIR
            elif (
                colonies[row, column] == zero_colonies[row, column]
                and float(block[row, column]) == 0.0
            ):
                rule = b.RULE_CONTRADICTION
            elif pair in served:
                rule = b.RULE_SERVED
            else:
                rule = "stored"
            pair_rule.setdefault(pair, set()).add(rule)
            if rule == "stored":
                kept += 1
                stored_pairs.add(pair)
            else:
                rules[rule] += 1
    released_cells = len(scores.recipient_tags) * len(scores.query_tags)
    dropped = sum(rules.values())
    if dropped + kept != released_cells:
        raise RuntimeError("the rules do not account for every released cell")
    return {
        "released_cells": released_cells,
        "kept_records": kept,
        "kept_oriented_pairs": len(stored_pairs),
        "dropped_records": dropped,
        "dropped_by_rule": {rule: rules[rule] for rule in b.DROP_RULES},
        "pair_rule": pair_rule,
        "stored_pairs": stored_pairs,
    }


def locate_served_pairs(
    butland_pairs: set[tuple[str, str]], pair_rule: dict[tuple[str, str], set[str]]
) -> dict[str, Any]:
    """Classify each served Butland-screen pair by where it sits in the release."""
    by_rule: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    absent: list[str] = []
    for pair in sorted(butland_pairs):
        label = f"{pair[0]} -> {pair[1]}"
        if pair not in pair_rule:
            by_rule["not_a_cell_of_this_release"] += 1
            absent.append(label)
            continue
        for rule in sorted(pair_rule[pair]):
            by_rule[rule] += 1
            examples.setdefault(rule, []).append(label)
    return {
        "n_served_butland_pairs": len(butland_pairs),
        "by_first_rule_that_applies": dict(sorted(by_rule.items())),
        "pairs_not_a_cell_of_this_release": absent,
        "examples_by_rule": {
            rule: items[:5] for rule, items in sorted(examples.items())
        },
    }


def stored_side(store_root: str) -> dict[str, Any] | None:
    """The built Butland store's content ids and oriented pairs, or None if absent."""
    if not osp.isdir(osp.join(store_root, "processed", "lmdb")):
        return None
    from torchcell.verification.runners import stream_records

    content_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    records = 0
    for record in stream_records(store_root):
        experiment = record["experiment"]
        by_cassette = {
            str(leaf["cassette"]): str(leaf["systematic_gene_name"])
            for leaf in experiment["genotype"]["perturbations"]
        }
        pairs.add((by_cassette[QUERY_CASSETTE], by_cassette[RECIPIENT_CASSETTE]))
        content_ids.add(
            hashlib.sha256(json.dumps(experiment).encode("utf-8")).hexdigest()
        )
        records += 1
    return {
        "store_root": store_root,
        "records": records,
        "content_ids": content_ids,
        "oriented_pairs": pairs,
    }


def main() -> int:
    """Measure both sides, write the record, and print the numbers the pins take."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    butland_root = osp.join(data_root, DATASET_ROOT_REL)
    served = served_side(osp.join(data_root, BABU_ROOT_REL))
    butland_pairs = {
        pair for pair, screen in served["pairs"].items() if screen == BABU_SCREEN_TAG
    }
    release = release_side(osp.join(butland_root, "raw"))
    counts = ledger(release, served["pairs"])
    located = locate_served_pairs(butland_pairs, counts["pair_rule"])
    stored = stored_side(butland_root)
    if stored is None:
        overlap: dict[str, Any] = {
            "measured": False,
            "reason": "the Butland dev store holds no processed/lmdb, so its content "
            "ids cannot be read; rebuild it and re-run",
        }
    else:
        shared_ids = stored["content_ids"] & served["content_ids"]
        shared_pairs = stored["oriented_pairs"] & set(served["pairs"])
        overlap = {
            "measured": True,
            "butland_records": stored["records"],
            "butland_content_ids": len(stored["content_ids"]),
            "babu_records": served["records"],
            "babu_content_ids": len(served["content_ids"]),
            "shared_content_ids": len(shared_ids),
            "shared_oriented_pairs": len(shared_pairs),
            "shared_examples": sorted(shared_ids)[:5],
        }
    record = {
        "served_babu2014": {
            key: value
            for key, value in served.items()
            if key not in ("pairs", "content_ids")
        },
        "release_butland2008": {
            "released_cells": counts["released_cells"],
            "kept_records": counts["kept_records"],
            "kept_oriented_pairs": counts["kept_oriented_pairs"],
            "dropped_records": counts["dropped_records"],
            "dropped_by_rule": counts["dropped_by_rule"],
            "n_queries": len(set(release["scores"].query_tags)),
            "n_recipient_rows": len(release["scores"].recipient_tags),
        },
        "served_butland_screen_pairs_located": located,
        "nothing_stored_twice": overlap,
        "pins": {
            "SERVED_BUTLAND_RECORDS": len(butland_pairs),
            "SERVED_BUTLAND_PAIRS": len(butland_pairs),
            "SERVED_OVERLAP_CELLS": counts["dropped_by_rule"][b.RULE_SERVED],
            "EXPECTED_RECORDS": counts["kept_records"],
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "butland2008_babu2014_partition.json")
    with open(out, "w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    print(json.dumps(record, indent=2, default=_json_default))
    print(f"wrote {out}")
    return 0


def _json_default(value: Any) -> Any:
    """Serialize the numpy scalars the matrix arithmetic produces."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"{type(value)!r} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
