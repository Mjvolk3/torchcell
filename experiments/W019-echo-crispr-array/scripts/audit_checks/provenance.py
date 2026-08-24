# experiments/W019-echo-crispr-array/scripts/A4-checks.py  (splice into verify_triple_build_list.py)
"""Provenance checks for the five pinned inputs the trigenic round consumes.

The design and the run-4 record read five files that no other check in the verifier
looks at as FILES: it reads their contents, never their identity. So an upstream
regeneration that silently changes a prediction, a fitness or a published reference
would leave every existing check green. These pin identity: sha256, row and column
shape, the columns the design actually consumes, and the derived 39-target basis.

Two of the checks also settle a documented ambiguity rather than only detecting drift:

  * `top_k_constructible_panel12_k200.csv` (52 rows) is a strict subset of
    `triples_table_panel12_k200.csv` (122 rows). Same inference run, same 12-gene panel,
    different filter: the 52 are the panel-internal members of the genome-wide top 200,
    the 122 are every panel-internal triple present in the inference table at all. The
    audit brief's "different gene panel" label does not apply to that file.
  * `results/constructible_triples_panel12_k200.parquet` (no `inference_3/` in the path)
    IS a different panel: 12 genes of which only YPL046C is in the design basis. That is
    the file the "do not use" note is about.

Splicing contract: `run(check)` is the only public entry point. `check` is the verifier's
existing recorder, `check(group, ok, claim, observed)`. Every check uses group
"provenance". The module-level path block below duplicates the verifier's own
DATA_ROOT / EXP_DIR / REPO / RESULTS / NOTES / STRAINS / TARGETS definitions verbatim so
this file also runs standalone; drop it when splicing and the names resolve to the
verifier's.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import os.path as osp

import pandas as pd
from dotenv import find_dotenv, load_dotenv

# ---- stands in for verify_triple_build_list.py's own path block; DROP ON SPLICE --------
# Same values, resolved from EXPERIMENT_ROOT instead of __file__ so this file also runs
# from outside the scripts directory. Spliced in, these names come from the verifier.
# usecwd=True: this file may sit outside the repo before it is spliced, so resolve .env
# from the working directory (repo root, per the project's run convention), not __file__.
load_dotenv(find_dotenv(usecwd=True))
DATA_ROOT = os.environ["DATA_ROOT"]
EXP_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "W019-echo-crispr-array")
REPO = osp.dirname(os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXP_DIR, "results")
NOTES = osp.join(REPO, "notes")
STRAINS = osp.join(
    EXP_DIR, "data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv"
)
TARGETS = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/inference_3",
    "top_k_constructible_panel12_k200.csv",
)
# ---------------------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(check) -> None:
    """Add the `provenance` group to the verifier. Cheap, deterministic, file-only."""
    # ---- NEW constants (all local to run()) -------------------------------------------
    BOOTSTRAP = osp.join(RESULTS, "run4_strain_bootstrap.csv")
    REF_SMF = osp.join(RESULTS, "reference_smf_12panel.csv")
    DOUBLES_REF = osp.join(
        REPO, "experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv"
    )
    SHEET_MANIFEST = osp.join(
        EXP_DIR, "data/run4_doubles_2026-08-06/SHA256SUMS.txt"
    )
    TRIPLES_TABLE_122 = osp.join(
        REPO, "experiments/010-kuzmin-tmi/results/inference_3",
        "triples_table_panel12_k200.csv",
    )
    PARQUET_OTHER_PANEL = osp.join(
        REPO, "experiments/010-kuzmin-tmi/results",
        "constructible_triples_panel12_k200.parquet",
    )
    # sha256 recorded 2026-08-24 by audit A4. A mismatch means the upstream artifact was
    # regenerated; that is a provenance event to record, not a number to update in place.
    PINNED_SHA256 = {
        TARGETS: "3bfd68b3d2d89b3c7416412039aa0d71b020a74be87763b0678440e1e7fbb855",
        BOOTSTRAP: "31c941ff86e5f404e76d65e2701dfb26d5ee36bcc9679ab043f72f74dd337d11",
        REF_SMF: "484fa3502e898f230ca5be70381c8ef64b581259d3159b289f4f979130c8e3f1",
        DOUBLES_REF: "d70cc9c361804c87be7bc2e20c381a1d390ac398136964713649b66f8e0822ce",
        STRAINS: "cc0f07a7e7123aa0b1088562bf6781a4e7f443635932d9461f7bcaf64628d8ab",
    }
    PINNED_SHAPE = {
        TARGETS: (52, 4),
        BOOTSTRAP: (26, 5),
        REF_SMF: (12, 7),
        DOUBLES_REF: (45, 15),
        STRAINS: (26, 3),
    }
    TARGET_COLS = ["gene1", "gene2", "gene3", "prediction"]
    BASIS = frozenset((
        "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
        "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
    ))
    TEN = BASIS - {"YLR104W"}          # the frozen set the original set-cover ran on
    N_PLATES = 3                       # run 4 plated P1/P2/P3, all three QC-pass
    # A nonparametric bootstrap of the mean of n values has SE = sd_sample*sqrt(n-1)/n,
    # so boot_se / across_plate_sd = sqrt(n-1)/n. At n=3 that is 0.4714. Reproducing it
    # from the two stored columns is what shows boot_se IS a bootstrap SE of the plate
    # mean and across_plate_sd IS the ddof=1 sample SD across plates.
    BOOT_SD_RATIO = (N_PLATES - 1) ** 0.5 / N_PLATES
    BOOT_SD_TOL = 0.02
    # -----------------------------------------------------------------------------------

    label = {TARGETS: "top_k_constructible_panel12_k200.csv",
             BOOTSTRAP: "run4_strain_bootstrap.csv",
             REF_SMF: "reference_smf_12panel.csv",
             DOUBLES_REF: "construction_validation_doubles.csv",
             STRAINS: "Single-and-Double-KO-Strains-List-Order.csv"}

    for path, want in PINNED_SHA256.items():
        got = _sha256(path)
        check("provenance", got == want,
              f"{label[path]} sha256 unchanged since 2026-08-24", got[:16])

    for path, want in PINNED_SHAPE.items():
        got = pd.read_csv(path).shape
        check("provenance", got == want,
              f"{label[path]} is {want[0]} rows x {want[1]} columns", got)

    # The bench sheet is hand-maintained and has NO generating script, so its own shipped
    # manifest is the only thing that can catch a bad edit. Check the two agree.
    manifest = dict(
        (line.split()[1], line.split()[0])
        for line in open(SHEET_MANIFEST, encoding="utf-8").read().splitlines() if line.strip()
    )
    sheet_name = osp.basename(STRAINS)
    check("provenance", manifest[sheet_name] == _sha256(STRAINS),
          "the hand-maintained bench sheet matches its own SHA256SUMS.txt entry",
          manifest[sheet_name][:16])

    # ---- the prediction file: columns, contents, derived basis -------------------------
    tgt = pd.read_csv(TARGETS)
    check("provenance", list(tgt.columns) == TARGET_COLS,
          "the k200 file carries exactly the four columns the design reads",
          list(tgt.columns))
    tri = [frozenset(r) for r in tgt[["gene1", "gene2", "gene3"]].to_numpy()]
    check("provenance", len(tri) == len(set(tri)) and all(len(t) == 3 for t in tri),
          "every k200 row is a distinct 3-gene triple", f"{len(set(tri))} distinct")
    genes = set().union(*tri)
    check("provenance", len(genes) == 12 and BASIS < genes,
          "the k200 file spans a 12-gene panel that strictly contains the 11-gene basis",
          sorted(genes - BASIS))
    in_basis = [t for t in tri if t <= BASIS]
    check("provenance", len(in_basis) == 39,
          "39 of the 52 k200 triples lie inside the 11-gene basis", len(in_basis))
    check("provenance",
          sum(1 for t in in_basis if "YLR104W" in t) == 8
          and sum(1 for t in in_basis if t <= TEN) == 31,
          "YLR104W contributes 8 of the 39; the frozen TEN set accounts for the other 31",
          f"{sum(1 for t in in_basis if 'YLR104W' in t)} + "
          f"{sum(1 for t in in_basis if t <= TEN)}")
    check("provenance", tgt["prediction"].is_monotonic_decreasing,
          "the k200 file is already sorted by descending prediction",
          f"{tgt.prediction.max():.4f} .. {tgt.prediction.min():.4f}")

    # ---- adjudicate the two "do not use" artifacts -------------------------------------
    tt = pd.read_csv(TRIPLES_TABLE_122)
    tt_tri = {frozenset(r) for r in tt[["gene1", "gene2", "gene3"]].to_numpy()}
    tt_genes = set().union(*tt_tri)
    check("provenance",
          len(tt) == 122 and set(tri) < tt_tri and tt_genes == genes,
          "the 122-row triples table is the SAME panel and a strict superset of k200, "
          "not a different panel",
          f"{len(tt)} rows, {len(tt_genes)} genes, k200 subset={set(tri) < tt_tri}")
    pq = pd.read_parquet(PARQUET_OTHER_PANEL)
    pq_genes = set(pq.gene1) | set(pq.gene2) | set(pq.gene3)
    check("provenance", pq_genes & BASIS == {"YPL046C"},
          "results/constructible_triples_panel12_k200.parquet is a DIFFERENT gene panel, "
          "sharing one gene with the design basis",
          sorted(pq_genes & BASIS))

    # ---- the measured-fitness bootstrap -------------------------------------------------
    boot = pd.read_csv(BOOTSTRAP)
    check("provenance", set(boot.n_plates) == {N_PLATES},
          f"every run-4 strain was scored on all {N_PLATES} plates",
          sorted(set(boot.n_plates)))
    mut = boot[boot.across_plate_sd > 0]
    ratio = (mut.boot_se / mut.across_plate_sd)
    check("provenance", bool(((ratio - BOOT_SD_RATIO).abs() < BOOT_SD_TOL).all()),
          f"boot_se/across_plate_sd sits at sqrt(n-1)/n = {BOOT_SD_RATIO:.4f}, so boot_se "
          "is a bootstrap SE of the plate mean and across_plate_sd a sample SD",
          f"{ratio.min():.4f} .. {ratio.max():.4f} over {len(ratio)} strains")
    check("provenance", float(boot.loc[boot.strain == "WT", "fitness"].iloc[0]) == 1.0,
          "WT is the within-plate normalizer, so its fitness is exactly 1",
          float(boot.loc[boot.strain == "WT", "fitness"].iloc[0]))

    # ---- the two published reference tables ---------------------------------------------
    ref = pd.read_csv(REF_SMF)
    # This check previously asserted that the reference had NO row for YLR104W, encoding
    # the defect as the expected state: the panel still carried LCL1 (YPL056C) from the
    # pre-run-4 design while run 4 kept LCL2. Corrected 2026.08.24, build_reference_smf.py
    # now lists YLR104W and Costanzo covers all 12.
    check("provenance",
          int(ref.costanzo_smf.notna().sum()) == 12
          and "YLR104W" in set(ref.orf) and "YPL056C" not in set(ref.orf),
          "the SMF reference covers 12 of 12 and carries YLR104W (LCL2), the gene run 4 "
          "actually plated, not LCL1",
          f"{int(ref.costanzo_smf.notna().sum())}/12 Costanzo, "
          f"YLR104W present={'YLR104W' in set(ref.orf)}, "
          f"YPL056C present={'YPL056C' in set(ref.orf)}")
    dref = pd.read_csv(DOUBLES_REF)
    want_pairs = {frozenset(p) for p in itertools.combinations(sorted(TEN), 2)}
    got_pairs = {frozenset((r.gene1, r.gene2)) for r in dref.itertuples()}
    check("provenance", got_pairs == want_pairs,
          "the doubles reference covers exactly the 45 within-TEN pairs, C(10,2)",
          f"{len(got_pairs)} pairs")


if __name__ == "__main__":
    TALLY = {"n": 0, "fail": 0}

    def _check(group: str, ok: bool, claim: str, observed: object) -> None:
        TALLY["n"] += 1
        TALLY["fail"] += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  [{group}] {claim}  [{observed}]")

    run(_check)
    print(f"\n{TALLY['n']} checks, {TALLY['fail']} FAIL")
    raise SystemExit(1 if TALLY["fail"] else 0)
