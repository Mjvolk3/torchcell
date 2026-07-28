# experiments/020-cachera-betaxanthin/scripts/build_merzbacher_split.py
# [[experiments.020-cachera-betaxanthin.scripts.build_merzbacher_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/build_merzbacher_split
"""Nest Merzbacher 2025's betaxanthin test split inside our genome-scale split.

WHY NESTING RATHER THAN SHRINKING
---------------------------------
The usual way to compare against a published model is to shrink your data to theirs. That
throws away the thing we actually claim: Merzbacher's Flux Cone Learning can only make a
prediction for the ~811 deletions that are Yeast9 metabolic genes, because a gene outside the
GEM has no flux cone. We predict all ~4,700 in the Cachera screen.

So we constrain **evaluation**, not training. Their 640 test genes are forced into OUR test
split; everything else in the screen goes to train/val. We then train genome-scale and score
on exactly their genes. The comparison happens on their turf while the model keeps the
genome-scale signal that is the point.

THE SPLIT IS RELEASED -- THE PAPER JUST DOESN'T SAY SO
------------------------------------------------------
The manuscript reports no split and an internally inconsistent test N (Fig. 4b says 659,
Table S6 says 649, and 20 % of 811 is 162). The Zenodo **code** deposit
(10.5281/zenodo.15518666 -> record 15761895) contains the exact partition:

  data/yeast_production_test_split.csv        640 systematic ORFs -- the test set
  data/yeast_production_validation_split.csv  the same 640 with `label` 0/1/2 and `fold` 0-4
  training/yeast_training_production.py       the training loop

Only the 49 MB code archive is fetched; `data.zip` is 23 GB of flux samples we do not need.

AND SO ARE THE THRESHOLDS. The paper calls its binning "qualitative"; the code is exact
(`yeast_training_production.py:47-53`) and is reproduced verbatim in :func:`merzbacher_bins`:
min-max scale production to [0, 1], then cut at **0.40** and **0.65**.

That last point carries a trap worth stating loudly. **Min-max scaling makes the thresholds
depend on the observed extremes**, so applying 0.40/0.65 to a differently-ranged copy of the
screen silently produces different classes. Our build has 4,735 deletions to their 4,223, so
the min and max are not guaranteed to match. This script therefore reports the class
distribution BOTH ways -- our-range and their-range -- and the agreement between our labels and
their released labels on the shared genes. If those disagree, the head-to-head is comparing
different tasks and the number must not be quoted.

TWO RECONCILIATIONS THIS SCRIPT REPORTS RATHER THAN ASSUMES
------------------------------------------------------------
1. **640 vs 811.** The paper says 811 Yeast9 metabolic genes; the released test list is 640.
   The Methods mention deletions "where the sampling failed to converge" without counting
   them. We report the gap, we do not explain it away.
2. **Their validation set appears to sit inside their test set.** In `main()`, `val_names` are
   drawn from the same 640 genes as `test_names`, while `train_names` is built from the
   complement. If that reading is right, model selection touched test data. Confirming it
   needs `yeast_single_knockouts.npz` from the 23 GB archive, so this is FLAGGED, not claimed.
   It does not affect our nesting either way -- we only consume their gene list.

Output: ``results/merzbacher_nested_split.json`` (the split + every reconciliation count),
consumed by the trainer's forced-test-gene hook.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
import subprocess
import zipfile
from datetime import UTC, datetime
from typing import Any, Literal

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

#: Zenodo CONCEPT doi 10.5281/zenodo.15518666 resolves to this concrete record.
ZENODO_RECORD = "15761895"
CODE_ZIP = "deletionprediction-main.zip"
MIRROR_DIR = osp.join(DATA_ROOT, "data", "merzbacher2025_fcl")
CACHERA_LMDB = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/processed/lmdb"
)
CACHERA_RAW = osp.join(
    DATA_ROOT, "data/torchcell/betaxanthin_cachera2023/raw/GA1_2_4_6.csv"
)
#: The column the loader treats as the betaxanthin level (cachera2023.py:126).
CACHERA_LEVEL_COL = "corrected_mean_intensity.24_mean"
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results")

#: Verbatim from yeast_training_production.py:51. NOT re-derived -- copied.
MERZBACHER_THRESHOLDS = (0.40, 0.65)


class MerzbacherRetrieval(BaseModel):
    """Provenance for the fetched code archive.

    The stored zip plus its sha256 is canonical; the URL is retrieval metadata. On rebuild we
    replay ``retrieval_command`` and verify the hash, so upstream drift is DETECTED rather
    than silently followed.
    """

    source_url: str
    retrieval_method: Literal["zenodo_api"] = "zenodo_api"
    retrieval_command: str
    sha256: str
    retrieved_at: datetime
    bytes: int
    zenodo_record: str


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mirror_merzbacher_code() -> MerzbacherRetrieval:
    """Fetch + sha256-pin the code archive. Idempotent: verifies an existing mirror."""
    os.makedirs(MIRROR_DIR, exist_ok=True)
    zip_path = osp.join(MIRROR_DIR, CODE_ZIP)
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{CODE_ZIP}/content"
    command = f'curl -sL "{url}" -o {CODE_ZIP}'
    if not osp.exists(zip_path):
        subprocess.run(
            ["curl", "-sL", "--max-time", "600", url, "-o", zip_path], check=True
        )
    record = MerzbacherRetrieval(
        source_url=url,
        retrieval_command=command,
        sha256=_sha256_file(zip_path),
        retrieved_at=datetime.now(UTC),
        bytes=osp.getsize(zip_path),
        zenodo_record=ZENODO_RECORD,
    )
    with open(osp.join(MIRROR_DIR, "manifest.json"), "w") as f:
        f.write(record.model_dump_json(indent=2))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MIRROR_DIR)
    return record


def load_merzbacher_split() -> tuple[list[str], pd.DataFrame]:
    """Their 640 test ORFs, and the same genes with released label + CV fold."""
    base = osp.join(MIRROR_DIR, "deletionprediction-main", "data")
    test = pd.read_csv(osp.join(base, "yeast_production_test_split.csv"))["name"]
    val = pd.read_csv(osp.join(base, "yeast_production_validation_split.csv"))
    return list(test), val


def merzbacher_bins(
    values: np.ndarray, thresholds: tuple[float, float] = MERZBACHER_THRESHOLDS
) -> np.ndarray:
    """Their exact 3-class rule: min-max scale to [0,1], cut at 0.40 / 0.65.

    Reproduced from `yeast_training_production.py:47-53`. Note the scaling is over WHATEVER
    array is passed, which is precisely why the caller must decide -- and report -- whether
    the range is ours or theirs.
    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    scaled = (values - lo) / (hi - lo)
    t1, t2 = thresholds
    return np.where(scaled < t1, 0, np.where(scaled < t2, 1, 2))


def read_cachera_raw() -> tuple[dict[str, float], dict[str, str]]:
    """{systematic ORF -> betaxanthin} from the RAW screen, via the shared name resolver.

    WHY RAW AND NOT THE BUILT LMDB. A split is a list of gene names -- it does not need the
    built dataset, and building it from the LMDB would inherit that build's defects. The
    LMDB on disk is stale with respect to the layered name resolver (it predates commit
    567fa6aa and is in fact hardlinked in from the KG-build tree, uid 7474), so it is
    MISSING NINE of Merzbacher's test genes: RIP1, PRS3, SDH1, APT1, TSL1, PFK2, NRK1, MSF1,
    ANT1. All nine sit in the raw CSV with 11-16 colonies and no NaNs -- they were lost to
    name resolution, not to measurement.

    The raw file mixes conventions: only 1,107 of 4,788 rows carry a systematic ORF, the rest
    use common names. That is exactly what ``resolve_gene_name`` exists for, and resolving
    here recovers the nine without a dataset rebuild (which would mean re-running the KG
    build and re-querying, far too heavy for a gene list).

    Returns the value map plus {raw name -> resolution status} so unresolved names are
    reported rather than silently dropped.
    """
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    df = pd.read_csv(CACHERA_RAW)
    # Roots pinned to DATA_ROOT. SCerevisiaeGenome's defaults are RELATIVE, so a bare
    # constructor writes a genome tree into the caller's cwd and then tries to re-download
    # go.obo (which currently 403s) -- the same footgun already fixed in YeastGEM.
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    values: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for raw_name, level in zip(df["gene"], df[CACHERA_LEVEL_COL], strict=False):
        name = str(raw_name)
        if pd.isna(level):
            continue
        res = genome.resolve_gene_name(name)
        status = getattr(res.status, "value", str(res.status))
        statuses[name] = status
        # Same acceptance set as the loader (cachera2023.py:208-217): a RENAMED gene is a
        # real gene under an old label, not a bad record.
        if res.systematic_name is None or status not in {
            "current",
            "renamed",
            "non_gene_feature",
        }:
            continue
        values[res.systematic_name] = float(level)
    return values, statuses


def read_cachera_built() -> set[str]:
    """Genes actually present in the CURRENT build -- an availability check, not the split.

    The split is authoritative; this tells downstream how much of it is trainable today.
    """
    if not osp.exists(CACHERA_LMDB):
        return set()
    env = lmdb.open(CACHERA_LMDB, readonly=True, lock=False, readahead=False)
    out: set[str] = set()
    with env.begin() as txn:
        for _, raw in txn.cursor():
            exp = pickle.loads(raw)["experiment"]
            genes = [
                p["systematic_gene_name"]
                for p in exp["genotype"]["perturbations"]
                if p["perturbation_type"].endswith("deletion")
            ]
            if len(genes) == 1:
                out.add(genes[0])
    env.close()
    return out


def build() -> dict[str, Any]:
    """Emit the nested split plus every reconciliation count."""
    retrieval = mirror_merzbacher_code()
    their_test, their_val = load_merzbacher_split()
    ours, statuses = read_cachera_raw()
    built = read_cachera_built()

    their_set, our_set = set(their_test), set(ours)
    shared = sorted(their_set & our_set)
    missing = sorted(their_set - our_set)

    # OUR labels under THEIR rule, computed two ways -- see the min-max trap in the docstring.
    genes = sorted(our_set)
    vals = np.array([ours[g] for g in genes], dtype=float)
    labels_our_range = merzbacher_bins(vals)
    sub = np.array([ours[g] for g in shared], dtype=float)
    labels_their_range = merzbacher_bins(sub)

    # Agreement against their RELEASED labels on the shared genes.
    theirs = dict(zip(their_val["knockout"], their_val["label"], strict=False))
    idx = {g: i for i, g in enumerate(genes)}
    paired = [(theirs[g], int(labels_our_range[idx[g]])) for g in shared if g in theirs]
    agree = sum(a == b for a, b in paired)

    split = {
        "test": shared,  # their genes, restricted to ones we actually hold
        "train_val_pool": sorted(our_set - their_set),
    }
    report: dict[str, Any] = {
        "retrieval": json.loads(retrieval.model_dump_json()),
        "thresholds": list(MERZBACHER_THRESHOLDS),
        "reconciliation": {
            "their_test_genes": len(their_set),
            "paper_states_metabolic_genes": 811,
            "gap_640_vs_811": 811 - len(their_set),
            "our_single_deletion_genes": len(our_set),
            "shared": len(shared),
            "their_genes_missing_from_ours": len(missing),
            "missing_examples": missing[:10],
            "our_test_size": len(split["test"]),
            "our_train_val_pool": len(split["train_val_pool"]),
            "resolution_status_counts": {
                k: sum(1 for v in statuses.values() if v == k)
                for k in sorted(set(statuses.values()))
            },
        },
        "availability_in_current_build": {
            "genes_in_built_lmdb": len(built),
            "split_test_genes_trainable_today": len(set(split["test"]) & built),
            "split_test_genes_missing_from_build": sorted(set(split["test"]) - built),
            "note": (
                "The split is authoritative and derived from the RAW screen + the shared "
                "name resolver. The built LMDB is stale (predates resolver commit 567fa6aa, "
                "and is hardlinked from the KG tree, uid 7474), so it lacks some resolvable "
                "genes. Rebuilding needs a full KG rebuild + requery, so it is tracked as an "
                "issue rather than blocking this split."
            ),
        },
        "label_check": {
            "n_compared": len(paired),
            "agreement": agree / len(paired) if paired else None,
            "our_range_class_counts": {
                int(k): int(v)
                for k, v in zip(*np.unique(labels_our_range, return_counts=True), strict=False)
            },
            "their_range_class_counts": {
                int(k): int(v)
                for k, v in zip(*np.unique(labels_their_range, return_counts=True), strict=False)
            },
            "their_released_class_counts": their_val["label"].value_counts().sort_index().to_dict(),
            "note": (
                "Their rule min-max scales, so thresholds depend on the observed extremes. "
                "Low agreement here means our build and theirs induce DIFFERENT classes and "
                "the 3-class head-to-head must not be quoted."
            ),
        },
        "caveats": [
            "640 released test genes vs the paper's stated 811 -- gap unexplained by the paper.",
            "Their val folds are drawn from the same 640 genes as their test list; if that "
            "reading is right, their model selection touched test data. FLAGGED, not claimed: "
            "confirming needs yeast_single_knockouts.npz from the 23 GB data archive.",
        ],
        "split": split,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "merzbacher_nested_split.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    r = build()
    rec, lc = r["reconciliation"], r["label_check"]
    print("=" * 72)
    print("MERZBACHER NESTED SPLIT")
    print("=" * 72)
    print(f"  their test genes        {rec['their_test_genes']}")
    print(f"  paper states            {rec['paper_states_metabolic_genes']}  "
          f"(gap {rec['gap_640_vs_811']}, unexplained)")
    print(f"  our single-deletion set {rec['our_single_deletion_genes']}")
    print(f"  shared -> OUR TEST      {rec['our_test_size']}")
    print(f"  our train/val pool      {rec['our_train_val_pool']}")
    print(f"  their genes we lack     {rec['their_genes_missing_from_ours']} "
          f"{rec['missing_examples'][:5]}")
    print(f"\n  label agreement vs their released labels: "
          f"{lc['agreement']:.1%}" if lc["agreement"] is not None else "  no overlap")
    print(f"  our-range classes    {lc['our_range_class_counts']}")
    print(f"  their-range classes  {lc['their_range_class_counts']}")
    print(f"  their released       {lc['their_released_class_counts']}")
    print(f"\n  sha256 {r['retrieval']['sha256'][:32]}...")
    print(f"  wrote {osp.join(RESULTS_DIR, 'merzbacher_nested_split.json')}")


if __name__ == "__main__":
    main()
