# experiments/W019-echo-crispr-array/scripts/verify_plate_pairing.py
# [[experiments.W019-echo-crispr-array.scripts.verify_plate_pairing]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/verify_plate_pairing
"""Which picklist belongs to which plate image? Answer it from CONTENT, not from filenames.

`P2_OD1-5nL_TCsingleKO.JPG` is paired with `cherrypick_Plate2_384_5nL.csv` purely by a naming
convention. Nothing in either file proves the plate that was photographed as "P2" is the one
the Echo dispensed from Plate2's picklist. If two plates were swapped at imaging, every strain
label on them would be wrong -- and a wrong label set is exactly the kind of thing that
produces an inexplicable wild-type reference.

Two independent content-based discriminators, crossed over every (image, picklist,
orientation) combination:

  BLANKS  -- each picklist puts 6 `Blank_media` wells at DIFFERENT positions (verified), and
             media-only wells must be empty. Under a wrong pairing the blank positions land on
             occupied wells.
  STRAINS -- Kruskal-Wallis H across strain groups. Only the correct pairing makes a strain's
             29 replicates agree, so H peaks there. A wrong pairing turns every group into a
             mixture of all genotypes, collapsing H toward its null (~12 for 13 groups).

Also checks the blank GEOMETRY, which turned out to have a design flaw: the blanks are placed
at mirror-symmetric rows (r and 17-r), so a vertical flip maps blank positions onto blank
positions and the blank test cannot distinguish `identity` from `flip_v`. Columns are
asymmetric, so `rot180` and `flip_h` are excluded. Fixing this is a one-line change to the
next picklist generator: place blanks asymmetrically in BOTH axes.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/verify_plate_pairing.py
"""

from __future__ import annotations

import itertools
import os
import os.path as osp

import pandas as pd
import run2_volume_timepoints as r2
from dotenv import load_dotenv
from scipy.stats import kruskal

from torchcell.sga import NormalizationConfig, normalize_plate, read_echo_picklist

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "W019-echo-crispr-array")
DATA = osp.join(EXP_DIR, "data", "run3_2026-07-23")
QUANT = osp.join(EXP_DIR, "quant", "run3_proc")
RESULTS = osp.join(EXP_DIR, "results")

N_ROWS, N_COLS = 16, 24
BLANK_NAME = "Blank_media"
IMAGES = ["P1", "P2", "P3"]
PICKLISTS = {
    "Plate1": "cherrypick_Plate1_384_5nL.csv",
    "Plate2": "cherrypick_Plate2_384_5nL.csv",
    "Plate3": "cherrypick_Plate3_384_5nL.csv",
}
# `identity` means row 1 / col 1 sits at the TOP-LEFT of the cropped image, i.e. well A1 top
# left -- the convention the plates are photographed in.
EXPECTED_ORIENTATION = "identity"


def blank_geometry() -> pd.DataFrame:
    """Are the blank wells asymmetric enough to pin the orientation on their own?"""
    rows = []
    for name, f in PICKLISTS.items():
        lay = read_echo_picklist(osp.join(DATA, f))
        b = lay[lay.strain == BLANK_NAME][["row", "col"]]
        pos = set(map(tuple, b.to_numpy().tolist()))
        rows.append({
            "picklist": name,
            "blank_wells": sorted(pos),
            # a flip maps (r, c) -> (N_ROWS + 1 - r, c) etc.; if the mapped set equals the
            # original set, that flip is INVISIBLE to the blank test
            "vflip_invariant": {(N_ROWS + 1 - r, c) for r, c in pos} == pos,
            "hflip_invariant": {(r, N_COLS + 1 - c) for r, c in pos} == pos,
        })
    return pd.DataFrame(rows)


def cross_pair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every (image, picklist) pair, maximised over the 4 orientations."""
    cfg = NormalizationConfig()
    grids = {g: pd.read_csv(osp.join(QUANT, f"run3_grid_{g}.csv")) for g in IMAGES}
    lays = {k: read_echo_picklist(osp.join(DATA, f)) for k, f in PICKLISTS.items()}

    blanks = pd.DataFrame(index=IMAGES, columns=list(PICKLISTS), dtype=int)
    hstat = pd.DataFrame(index=IMAGES, columns=list(PICKLISTS), dtype=float)
    orient = pd.DataFrame(index=IMAGES, columns=list(PICKLISTS), dtype=object)

    for img, pk in itertools.product(IMAGES, PICKLISTS):
        best = (-1, -1.0, "")
        for op in r2.OPS:
            m = r2.apply_orientation(grids[img], op).merge(
                lays[pk], on=["row", "col"], how="inner"
            )
            is_blank = m.strain == BLANK_NAME
            n_empty = int((is_blank & (m["size"] <= cfg.min_size)).sum())
            df = normalize_plate(m, cfg)
            ok = df[~df.is_missing & ~df.is_flagged & ~df.is_blank & ~df.is_jackknife]
            groups = [v["norm"].dropna().to_numpy() for _, v in ok.groupby("strain")
                      if len(v) >= 5]
            h = float(kruskal(*groups).statistic) if len(groups) > 2 else float("nan")
            if (n_empty, h) > (best[0], best[1]):
                best = (n_empty, h, op)
        blanks.loc[img, pk], hstat.loc[img, pk], orient.loc[img, pk] = best
    return blanks, hstat, orient


def main() -> None:
    """Verify the pairing and fail loudly if the filename convention is not what the data says."""
    os.makedirs(RESULTS, exist_ok=True)

    geom = blank_geometry()
    print("[1] blank-well geometry -- can blanks alone pin the orientation?")
    for _, r in geom.iterrows():
        print(f"    {r.picklist}: {r.blank_wells}")
        print(f"      invariant under vertical flip: {r.vflip_invariant}"
              f"   under horizontal flip: {r.hflip_invariant}")
    if geom.vflip_invariant.any():
        print("    DESIGN FLAW: blanks sit at mirror rows (r and 17-r), so a vertical flip")
        print("    maps blanks onto blanks and the blank test CANNOT rule out flip_v.")
        print("    Fix for the next picklist: place blanks asymmetrically in BOTH axes.")

    blanks, hstat, orient = cross_pair()
    print("\n[2] blank wells empty (max 6), best over orientations")
    print(blanks.to_string())
    print("\n[3] Kruskal-Wallis H across strains, best over orientations")
    print("    (null ~12 for 13 groups; large H means replicates agree = correct pairing)")
    print(hstat.astype(float).round(1).to_string())
    print("\n[4] orientation chosen per cell")
    print(orient.to_string())

    print("\n[5] every possible assignment, ranked by total H")
    ranked = sorted(
        (
            (sum(float(hstat.iloc[i, perm[i]]) for i in range(len(IMAGES))), perm)
            for perm in itertools.permutations(range(len(IMAGES)))
        ),
        reverse=True,
    )
    for score, perm in ranked:
        label = " ".join(f"{IMAGES[i]}->{list(PICKLISTS)[perm[i]]}" for i in range(len(IMAGES)))
        print(f"    {score:>8.1f}   {label}")
    best_score, best_perm = ranked[0]

    assert best_perm is not None
    runner_up = max(
        sum(float(hstat.iloc[i, p[i]]) for i in range(len(IMAGES)))
        for p in itertools.permutations(range(len(IMAGES))) if p != best_perm
    )
    name_match = all(best_perm[i] == i for i in range(len(IMAGES)))
    diag_blanks = all(int(blanks.iloc[i, i]) == 6 for i in range(len(IMAGES)))
    diag_orient = all(orient.iloc[i, i] == EXPECTED_ORIENTATION for i in range(len(IMAGES)))

    print("\n[6] verdict")
    print(f"    best assignment matches the filename convention : {name_match}")
    print(f"    margin over the next-best assignment            : {best_score:.1f} vs {runner_up:.1f}")
    print(f"    all diagonal pairs place 6/6 blanks on empty wells: {diag_blanks}")
    print(f"    all plates resolve to '{EXPECTED_ORIENTATION}' (A1 top-left): {diag_orient}")

    out = pd.DataFrame({
        "image": IMAGES,
        "picklist": [list(PICKLISTS)[best_perm[i]] for i in range(len(IMAGES))],
        "blanks_empty": [int(blanks.iloc[i, best_perm[i]]) for i in range(len(IMAGES))],
        "kruskal_h": [float(hstat.iloc[i, best_perm[i]]) for i in range(len(IMAGES))],
        "orientation": [orient.iloc[i, best_perm[i]] for i in range(len(IMAGES))],
        "matches_filename": name_match,
    })
    out.to_csv(osp.join(RESULTS, "run3_pairing_verification.csv"), index=False)
    hstat.astype(float).to_csv(osp.join(RESULTS, "run3_pairing_cross_h.csv"))
    print("\nwrote", osp.join(RESULTS, "run3_pairing_verification.csv"))

    if not (name_match and diag_blanks and diag_orient):
        raise SystemExit("PAIRING CHECK FAILED -- do not trust the filename convention here.")
    print("PASS: the filename convention is confirmed by the data.")


if __name__ == "__main__":
    main()
