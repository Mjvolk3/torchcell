# experiments/018-natural-isolate-genomics/scripts/audit_caudal_missing_absences.py
# [[experiments.018-natural-isolate-genomics.scripts.audit_caudal_missing_absences]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/audit_caudal_missing_absences

"""caudal2024 silently omits ~133 gene-absence edits per isolate. Quantify it.

``CaudalPanTranscriptome2024Dataset._content_perturbations`` takes ``s288c_mask`` as a
parameter and **never uses it**. Both loops guard on ``core_mask``
(``presence.mean(axis=0) >= CORE_PRESENCE_THRESHOLD``, i.e. 0.99):

    presence : (~core_mask) & (presence_row == 1)   # non-core ORF PRESENT -> emit
    absence  :   core_mask  & (presence_row == 0)   # core ORF ABSENT      -> emit

An S288C **reference ORF that is variable (not core) and ABSENT from the isolate** matches
NEITHER loop, so **no perturbation record is emitted at all** and the isolate reconstructs
as if it still carries the gene. The function's own docstring promises the opposite --
"Every absence is recorded ... never dropped (a dropped absence would wrongly reconstruct
as present)".

The guard is on the wrong axis. Presence/absence relative to S288C is a question about
**reference membership** (is this ORF in S288C?), not about **population frequency** (is
this ORF in >=99% of isolates?). Those are different sets: 5,491 ORFs are "core" by
frequency, 6,069 are true S288C reference ORFs, and the two only partially overlap.

``s288c_mask`` -- the mask that SHOULD gate the absence loop -- is additionally
INCOMPLETE: it is built from ``_orf_to_s288c``, which returns ``None`` for all 804
``_NumOfGenes_N`` paralog-cluster columns, of which **793 are real reference ORFs**
(YAL005C/SSA1, YAL038W/CDC19, ...). So any fix must strip that suffix before mapping.

This script quantifies the gap against the released presence/absence matrix. Fixing the
loader requires a Caudal LMDB rebuild.
"""

import gzip
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics", "results")

PRESENCE = osp.join(
    DATA_ROOT,
    "torchcell-library/peterGenomeEvolution10112018/data",
    "genesMatrix_PresenceAbsence.tab.gz",
)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from torchcell.datasets.scerevisiae.caudal2024 import (
        CORE_PRESENCE_THRESHOLD,
        _demangle_orf,
        _orf_to_s288c,
    )

    with gzip.open(PRESENCE, "rt") as fh:
        pa = pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)

    orf_ids = [_demangle_orf(c) for c in pa.columns]
    vals = pa.to_numpy()

    # exactly the masks the loader builds (caudal2024.py:295, :298)
    s288c_mask = np.array([_orf_to_s288c(o) is not None for o in orf_ids])
    core_mask = vals.mean(axis=0) >= CORE_PRESENCE_THRESHOLD

    # the mask that SHOULD gate absence: true reference membership, paralog clusters
    # recovered by stripping the _NumOfGenes_N suffix before mapping
    true_ref = np.array(
        [bool(_orf_to_s288c(re.sub(r"_NumOfGenes_\d+$", "", o))) for o in orf_ids]
    )

    absent = vals == 0
    emitted = (core_mask[None, :] & absent).sum(axis=1)  # what the loader records
    should = (true_ref[None, :] & absent).sum(axis=1)  # what it should record
    missing = ((true_ref & ~core_mask)[None, :] & absent).sum(axis=1)  # the gap

    print(f"CORE_PRESENCE_THRESHOLD = {CORE_PRESENCE_THRESHOLD}")
    print(f"pangenome ORFs                        : {len(orf_ids)}")
    print(
        f"  core_mask   (frequency >= 0.99)     : {int(core_mask.sum())}  "
        f"<- what the loops actually use"
    )
    print(f"  s288c_mask  (computed, NEVER USED)  : {int(s288c_mask.sum())}")
    print(
        f"  TRUE reference ORFs (incl. paralogs): {int(true_ref.sum())}  "
        f"(+{int(true_ref.sum() - s288c_mask.sum())} that s288c_mask misses)"
    )

    print(f"\n--- gene-absence records, per isolate (n={vals.shape[0]}) ---")
    for lbl, x in [
        ("loader EMITS      ", emitted),
        ("SHOULD emit       ", should),
        ("SILENTLY MISSING  ", missing),
    ]:
        print(
            f"  {lbl}: mean {x.mean():7.1f}  median {np.median(x):6.0f}  "
            f"max {int(x.max())}"
        )

    print(
        f"\n>>> {missing.mean():.0f} reference ORFs per isolate are ABSENT from the "
        f"genome but produce"
    )
    print(
        "    NO perturbation record -- the isolate reconstructs as if it HAS the gene."
    )
    print(f">>> {int(missing.sum()):,} missing gene-absence edits across the panel.")

    out = {
        "core_presence_threshold": CORE_PRESENCE_THRESHOLD,
        "n_pangenome_orfs": len(orf_ids),
        "n_core_mask": int(core_mask.sum()),
        "n_s288c_mask_computed_but_unused": int(s288c_mask.sum()),
        "n_true_reference_orfs": int(true_ref.sum()),
        "n_reference_orfs_missed_by_s288c_mask": int(true_ref.sum() - s288c_mask.sum()),
        "per_isolate": {
            "emitted_mean": float(emitted.mean()),
            "emitted_median": float(np.median(emitted)),
            "should_mean": float(should.mean()),
            "should_median": float(np.median(should)),
            "missing_mean": float(missing.mean()),
            "missing_median": float(np.median(missing)),
            "missing_max": int(missing.max()),
        },
        "total_missing_absence_edits": int(missing.sum()),
        "root_cause": (
            "_content_perturbations guards both loops on core_mask (population "
            "frequency) instead of s288c_mask (reference membership); s288c_mask is "
            "passed in but never referenced, and is itself incomplete because "
            "_orf_to_s288c returns None for all 804 _NumOfGenes_N paralog clusters"
        ),
        "fix_requires_lmdb_rebuild": True,
    }
    with open(osp.join(RESULTS_DIR, "caudal_missing_absences.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nresults -> {RESULTS_DIR}/caudal_missing_absences.json")


if __name__ == "__main__":
    main()
