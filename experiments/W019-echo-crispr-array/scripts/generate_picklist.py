# experiments/W019-echo-crispr-array/scripts/generate_picklist.py
# [[experiments.W019-echo-crispr-array.scripts.generate_picklist]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/generate_picklist
"""Generate randomized Echo cherry-pick lists with ORIENTATION-RESOLVING blank placement.

Earlier picklists (made outside this repo) placed the 6 `Blank_media` wells at mirror-symmetric
rows -- (5, 12), (4, 13), (6, 11), each pair summing to `n_rows + 1`. A vertical flip maps
`r -> n_rows + 1 - r`, so it maps that blank set exactly onto itself: the blanks are invisible
to the one flip they are meant to catch. `verify_plate_pairing.py` measured the consequence --
the blank-emptiness test scores a perfect 6/6 for BOTH `identity` and `flip_v` on every plate,
resolving only 2 of the 3 wrong orientations.

This generator fixes that. Blank wells are chosen so that for every non-identity orientation
`f`, `S and f(S)` is EMPTY -- every blank moves onto a well that should carry a colony. Six
blanks then pin the orientation on their own, with no reliance on strain structure.

Blanks are also spread by stratified sampling (one per block of the plate) so they probe
registration across the whole surface rather than one corner.

The output matches the Echo Cherry Pick format used previously, so it drops into the existing
workflow unchanged.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/generate_picklist.py
"""

from __future__ import annotations

import itertools
import os
import os.path as osp
import string

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "W019-echo-crispr-array")
OUT_DIR = osp.join(EXP_DIR, "picklists")

N_ROWS, N_COLS = 16, 24
WT_NAME, BLANK_NAME = "BY4741", "Blank_media"

# Source-plate map carried over from the existing 13-strain source plate (verified against
# cherrypick_Plate2_384_5nL.csv). New strains continue the odd-column pattern: C7, C9, ...
SOURCE_WELLS: dict[str, str] = {
    "BY4741": "A1", "YJR060W": "A3", "YPL081W": "A5", "YOS9": "A7", "ELC1": "A9",
    "COS111": "A11", "MMS2": "A13", "LCL1": "A15", "YEH1": "A17", "YER079W": "A19",
    "YKL033W-A": "A21", "SPH1": "A23", "YLR312C-B": "C1", "Blank_media": "C3",
}
SOURCE_PLATE, SOURCE_TYPE, DEST_TYPE = "Source[1]", "384PP_AQ_SP", "384PP_Dest"


def well_name(row: int, col: int) -> str:
    """(1-indexed row, col) -> Echo well label, e.g. (1, 1) -> 'A1'."""
    return f"{string.ascii_uppercase[row - 1]}{col}"


def orientation_maps() -> dict[str, callable]:  # noqa: F821
    """The three non-identity orientations a 16x24 plate can be photographed in."""
    return {
        "rot180": lambda r, c: (N_ROWS + 1 - r, N_COLS + 1 - c),
        "flip_v": lambda r, c: (N_ROWS + 1 - r, c),
        "flip_h": lambda r, c: (r, N_COLS + 1 - c),
    }


def blanks_break_all_symmetries(pos: set[tuple[int, int]]) -> bool:
    """True when every wrong orientation moves EVERY blank onto a non-blank well.

    `S and f(S) == empty` is stronger than `S != f(S)`: it guarantees the blank test loses
    all 6 wells under a wrong orientation, rather than merely some of them.
    """
    return all(
        not (pos & {f(r, c) for r, c in pos}) for f in orientation_maps().values()
    )


def choose_blank_wells(n_blanks: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Stratified, symmetry-breaking blank positions.

    The plate is cut into `n_blanks` blocks and one well is drawn per block, so the blanks
    probe registration across the whole surface. Draws repeat until the set breaks all three
    orientation symmetries.
    """
    n_bands = 2
    n_cols_blocks = n_blanks // n_bands
    row_edges = np.linspace(1, N_ROWS + 1, n_bands + 1).astype(int)
    col_edges = np.linspace(1, N_COLS + 1, n_cols_blocks + 1).astype(int)

    for _ in range(10_000):
        pos: set[tuple[int, int]] = set()
        for bi in range(n_bands):
            for ci in range(n_cols_blocks):
                r = int(rng.integers(row_edges[bi], row_edges[bi + 1]))
                c = int(rng.integers(col_edges[ci], col_edges[ci + 1]))
                pos.add((r, c))
        if len(pos) == n_blanks and blanks_break_all_symmetries(pos):
            return sorted(pos)
    raise RuntimeError("could not place blanks that break every orientation symmetry")


def build_plate(
    strains: dict[str, int], n_blanks: int, seed: int, plate_name: str, volume_nl: float
) -> pd.DataFrame:
    """One randomized 384-well plate as an Echo cherry-pick table.

    `strains` maps strain -> replicate count; the wild type absorbs whatever wells are left so
    the plate is packed with no unused positions.
    """
    rng = np.random.default_rng(seed)
    blanks = choose_blank_wells(n_blanks, rng)

    wells = [
        (r, c)
        for r, c in itertools.product(range(1, N_ROWS + 1), range(1, N_COLS + 1))
        if (r, c) not in set(blanks)
    ]
    labels: list[str] = []
    for strain, n in strains.items():
        labels += [strain] * n
    n_wt = len(wells) - len(labels)
    if n_wt < 0:
        raise ValueError(f"{len(labels)} strain wells requested but only {len(wells)} free")
    labels += [WT_NAME] * n_wt
    rng.shuffle(labels)

    rows = [
        {"row": r, "col": c, "strain": s} for (r, c), s in zip(wells, labels)
    ] + [{"row": r, "col": c, "strain": BLANK_NAME} for r, c in blanks]
    df = pd.DataFrame(rows).sort_values(["row", "col"]).reset_index(drop=True)

    missing = sorted(set(df.strain) - set(SOURCE_WELLS))
    if missing:
        raise KeyError(f"no source well recorded for: {missing}")

    return pd.DataFrame({
        "Source Plate Name": SOURCE_PLATE,
        "Source Plate Type": SOURCE_TYPE,
        "Source Well": df.strain.map(SOURCE_WELLS),
        "Sample Name": df.strain,
        "Destination Plate Name": plate_name,
        "Destination Plate Type": DEST_TYPE,
        "Destination Well": [well_name(r, c) for r, c in zip(df.row, df.col)],
        "Transfer Volume": volume_nl,
    })


def audit(pick: pd.DataFrame, label: str) -> dict[str, object]:
    """Re-derive the properties the pipeline depends on, from the emitted file."""
    rows = [string.ascii_uppercase.index(w[0]) + 1 for w in pick["Destination Well"]]
    cols = [int(w[1:]) for w in pick["Destination Well"]]
    lay = pd.DataFrame({"row": rows, "col": cols, "strain": pick["Sample Name"]})
    blanks = set(map(tuple, lay[lay.strain == BLANK_NAME][["row", "col"]].to_numpy().tolist()))
    kept = {
        name: len(blanks & {f(r, c) for r, c in blanks})
        for name, f in orientation_maps().items()
    }
    return {
        "plate": label,
        "n_wells": len(lay),
        "unique_wells": lay.assign(w=pick["Destination Well"]).w.nunique(),
        "n_strains": int(lay.strain.nunique()),
        "n_wt": int((lay.strain == WT_NAME).sum()),
        "n_blank": len(blanks),
        "blank_wells": sorted(blanks),
        **{f"blanks_surviving_{k}": v for k, v in kept.items()},
        "orientation_resolved_by_blanks": all(v == 0 for v in kept.values()),
    }


def main() -> None:
    """Emit the round-4 plates and audit every one before it leaves the script."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Round 4 placeholder: the 12 assayed single knockouts. Replace/extend this dict with the
    # 14 doubles once those strains exist -- the wild type absorbs the remaining wells, so the
    # plate stays packed whatever the panel size.
    panel = {s: 13 for s in SOURCE_WELLS if s not in (WT_NAME, BLANK_NAME)}

    audits = []
    for i in (1, 2, 3):
        name = f"Plate{i}_{len(panel)}strain_384_5nL"
        pick = build_plate(panel, n_blanks=6, seed=1000 + i, plate_name=name, volume_nl=5)
        path = osp.join(OUT_DIR, f"cherrypick_{name}.csv")
        pick.to_csv(path, index=False)
        a = audit(pick, name)
        audits.append(a)
        print(f"wrote {path}")

    print("\naudit -- do the blanks resolve the orientation on their own?")
    for a in audits:
        print(f"  {a['plate']}")
        print(f"    wells {a['n_wells']} (unique {a['unique_wells']}), strains {a['n_strains']},"
              f" WT {a['n_wt']}, blanks {a['n_blank']}")
        print(f"    blank wells: {a['blank_wells']}")
        print(f"    blanks still landing on blanks under rot180/flip_v/flip_h: "
              f"{a['blanks_surviving_rot180']}/{a['blanks_surviving_flip_v']}/"
              f"{a['blanks_surviving_flip_h']}")
        print(f"    orientation resolvable by blanks alone: "
              f"{a['orientation_resolved_by_blanks']}")

    pd.DataFrame(audits).to_csv(osp.join(OUT_DIR, "picklist_audit.csv"), index=False)
    if not all(a["orientation_resolved_by_blanks"] for a in audits):
        raise SystemExit("blank placement failed the symmetry audit")
    print("\nPASS: every plate's blanks break all three orientation symmetries.")

    # These files are PROVISIONAL: only the 12 assayed singles exist today, so the wild type
    # soaks up the wells the 14 doubles will occupy. Spell out the real round-4 composition so
    # the emitted plates are not mistaken for the final ones.
    n_next, reps, n_blanks = 26, 13, 6
    print(
        f"\nNOTE: provisional -- {len(panel)} strains defined, so WT absorbs "
        f"{audits[0]['n_wt']} wells. With the 14 doubles added: {n_next} strains x {reps} "
        f"reps = {n_next * reps}, + {n_blanks} blanks -> WT "
        f"{N_ROWS * N_COLS - n_next * reps - n_blanks} wells. Add the doubles to "
        f"SOURCE_WELLS and `panel`, then re-run."
    )


if __name__ == "__main__":
    main()
