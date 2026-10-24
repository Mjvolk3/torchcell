# experiments/043-expression-proteome-morphology/scripts/ontology_annotation_043.py
# [[experiments.043-expression-proteome-morphology.scripts.ontology_annotation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/ontology_annotation_043.py
"""The experiment ontology vocabulary of the 043 raw store, and every dataset's profile.

Reads EVERY record of every dataset in the raw store (``STORE_043_VERSION``; v2 =
``raw_043_v2``, 15 datasets), builds each record's
:func:`torchcell.data.experiment_ontology.annotation_recipe` (dataset name excluded by
construction) in DataLoader workers, builds the vocabulary from the union with
:func:`build_experiment_ontology_vocabulary` (no hand-named node), and writes

* the vocabulary JSON to ``$DATA_ROOT/.../043-expression-proteome-morphology/
  ontology_043<suffix>/experiment_ontology_vocab.json`` (read by ``train_043.py --stage
  prepare`` for prep_v4);
* ``results/ontology_annotation_043<suffix>.json``: the vocabulary, every dataset's
  distinct profiles (``describe_annotation``) with record counts, and the pairwise
  check: two datasets COLLIDE when some profile of one equals some profile of the
  other, i.e. the conditioning cannot tell those records apart.

An encoder check closes the run: identical annotations give identical tokens.

    STORE_043_VERSION=v2 python experiments/043-expression-proteome-morphology/scripts/ontology_annotation_043.py --workers 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import os.path as osp
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import torch

from torchcell.data.experiment_ontology import (
    AnnotationRecipe,
    OntologyDagEncoder,
    annotation_recipe,
    build_experiment_ontology_vocabulary,
    describe_annotation,
)

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from store_version_043 import active as active_store  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
VOCAB_NAME = "experiment_ontology_vocab.json"


def ontology_dir(data_root: str) -> str:
    """``$DATA_ROOT/data/torchcell/experiments/<EXPERIMENT>/ontology_043<suffix>``."""
    return osp.join(
        data_root,
        "data/torchcell/experiments",
        EXPERIMENT,
        f"ontology_043{active_store().suffix}",
    )


class _Recipes(torch.utils.data.Dataset[tuple[str, str]]):
    """``(dataset name, canonical recipe JSON)`` per raw record (runs in workers)."""

    def __init__(self, lmdb_path: str, items: list[tuple[str, int]]) -> None:
        self.lmdb_path = lmdb_path
        self.items = items
        self._env: Any = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[str, str]:
        import lmdb

        from torchcell.datamodels.schema import (
            EXPERIMENT_REFERENCE_TYPE_MAP,
            EXPERIMENT_TYPE_MAP,
        )

        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
            )
        name, raw_index = self.items[i]
        with self._env.begin() as txn:
            rec = json.loads(txn.get(f"data_{raw_index}".encode()))
        e = rec["experiment"]
        r = rec["experiment_reference"]
        exp = EXPERIMENT_TYPE_MAP[e["experiment_type"]](**e)
        ref = EXPERIMENT_REFERENCE_TYPE_MAP[r["experiment_reference_type"]](**r)
        if exp.dataset_name != name:
            raise ValueError(f"data_{raw_index}: {exp.dataset_name} under {name}")
        return name, recipe_key(annotation_recipe(exp, ref))


def recipe_key(recipe: AnnotationRecipe) -> str:
    """Canonical JSON of a recipe (the dedup key of a profile)."""
    return json.dumps(recipe.model_dump(), sort_keys=True)


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--max-per-dataset", type=int, default=None, help="debugging only")
    args = ap.parse_args()
    t0 = time.time()
    data_root = os.environ["DATA_ROOT"]
    gen = active_store()
    root = osp.join(data_root, "data/torchcell/experiments", EXPERIMENT, gen.raw_tag)
    with open(osp.join(root, "raw", "raw_dataset_name_index.json")) as f:
        index: dict[str, list[int]] = json.load(f)["index"]
    items = [(n, i) for n in sorted(index) for i in index[n][: args.max_per_dataset]]
    loader = torch.utils.data.DataLoader(
        _Recipes(osp.join(root, "raw", "lmdb"), items),
        batch_size=64,
        num_workers=args.workers,
        collate_fn=lambda b: b,
    )
    per_ds: dict[str, Counter[str]] = defaultdict(Counter)
    done = 0
    for chunk in loader:
        for name, key in chunk:
            per_ds[name][key] += 1
        done += len(chunk)
        if done % 20000 < 64:
            print(f"{done}/{len(items)} records", flush=True)
    keys = sorted({k for c in per_ds.values() for k in c})
    recipes = {k: AnnotationRecipe.model_validate_json(k) for k in keys}
    vocab = build_experiment_ontology_vocabulary(recipes=recipes.values())
    depths = vocab.depths()
    desc = {
        k: describe_annotation(vocab, vocab.vector_from_recipe(r))
        for k, r in recipes.items()
    }
    out_dir = ontology_dir(data_root)
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = osp.join(out_dir, VOCAB_NAME)
    vocab.to_json(vocab_path)
    datasets: dict[str, Any] = {}
    for name in sorted(per_ds):
        profiles = [
            {"n_records": n, "nodes": desc[k]} for k, n in per_ds[name].most_common()
        ]
        datasets[name] = {
            "n_records": int(sum(per_ds[name].values())),
            "n_profiles": len(profiles),
            "profiles": profiles,
        }
        print(
            f"{name}: {datasets[name]['n_records']} records, {len(profiles)} profile(s)"
        )
    collisions = []
    distinct = []
    for a, b in itertools.combinations(sorted(per_ds), 2):
        sa = {tuple(desc[k]) for k in per_ds[a]}
        sb = {tuple(desc[k]) for k in per_ds[b]}
        shared = sa & sb
        if shared:
            collisions.append(
                {
                    "a": a,
                    "b": b,
                    "shared_profiles": len(shared),
                    "identical_profile_sets": sa == sb,
                }
            )
        else:
            one_a, one_b = set(next(iter(sa))), set(next(iter(sb)))
            distinct.append(
                {"a": a, "b": b, "first_profile_difference": sorted(one_a ^ one_b)}
            )
    print(f"vocabulary: {len(vocab)} nodes, depth {max(depths)}; written {vocab_path}")
    print(f"{len(collisions)} colliding pairs of {len(collisions) + len(distinct)}:")
    for c in collisions:
        print(f"  {c['a']} ~ {c['b']} (identical sets: {c['identical_profile_sets']})")
    # encoder: identical annotations -> identical tokens, distinct -> distinct
    torch.manual_seed(0)
    enc = OntologyDagEncoder(vocab, state_dim=16, token_dim=40)
    x = torch.stack([vocab.vector_from_recipe(recipes[k]) for k in keys])
    with torch.no_grad():
        tok = enc(x)
    mode = "donot_use_mm_for_euclid_dist"
    xd = torch.cdist(x, x, compute_mode=mode)
    td = torch.cdist(tok, tok, compute_mode=mode)
    same_ok = bool((td[xd == 0] < 1e-5).all())
    diff_min = float(td[xd > 0].min()) if bool((xd > 0).any()) else float("nan")
    print(
        f"encoder: {len(keys)} distinct annotations; identical -> identical token: "
        f"{same_ok}; min token distance between distinct annotations {diff_min:.3g}"
    )
    result = {
        "store": gen.raw_tag,
        "n_records": done,
        "n_datasets": len(per_ds),
        "n_distinct_annotations": len(keys),
        "vocabulary_size": len(vocab),
        "vocabulary_depth": max(depths),
        "vocabulary_path": vocab_path,
        "vocabulary": vocab.model_dump(),
        "datasets": datasets,
        "colliding_pairs": collisions,
        "distinct_pairs": distinct,
        "encoder_check": {
            "identical_annotations_identical_tokens": same_ok,
            "min_token_distance_distinct": diff_min,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    res = osp.join(
        osp.dirname(SCRIPTS), "results", gen.result_name("ontology_annotation_043")
    )
    with open(res, "w") as f:
        json.dump(result, f, indent=1)
    print(f"written {res}")
    return 0 if same_ok else 1


if __name__ == "__main__":
    sys.exit(main())
