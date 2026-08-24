# experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py  (spliced section)
# [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]
"""Published-source checks for the W019 trigenic round: Kuzmin, Costanzo, SGD.

Splice `run` into `verify_triple_build_list.py` and call it from `main()` as
`run(check)`. It uses that module's DATA_ROOT, NOTES, RATIONALE and RESULTS, and defines
its own constants inside `run`. Every check is a file read, a sha256 of a mirrored
markdown file, or a lookup in a small CSV; nothing here re-scans an LMDB. The three CSVs
are produced by `published_source_audit.py`, which does the scanning once.

The `__main__` block at the bottom exists only so this file runs standalone; delete it
when splicing.
"""

from __future__ import annotations

import csv
import hashlib
import os.path as osp
import re


def run(check) -> None:
    print("\n9. PUBLISHED SOURCES (Kuzmin, Costanzo, SGD)")

    # -- new constants -------------------------------------------------------------------
    LIB = osp.join(DATA_ROOT, "torchcell-library")  # noqa: F821
    KUZMIN_SI = osp.join(LIB, "kuzminSystematicAnalysisComplex2018/si/si1.md")
    KUZMIN_SI_SHA = "2ec80d05d823976e12add17699ad759bcd983768d2f53e7eb6c0185b963b8291"
    COSTANZO_SI = osp.join(LIB, "costanzoGlobalGeneticInteraction2016/si/si1.md")
    COSTANZO_SI_SHA = "1828703b0ff739fdf1c0d9232fe4fd81a3ce95a1b111780f55ef63bfa676880e"
    GFF = osp.join(
        DATA_ROOT,  # noqa: F821
        "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830",
        "saccharomyces_cerevisiae_R64-4-1_20230830.gff",
    )
    HANDOFF = osp.join(NOTES, "experiments.W019-echo-crispr-array.run4-handoff.md")  # noqa: F821
    AUDIT_BRIEF = osp.join(NOTES, "experiments.W019-echo-crispr-array.audit-brief.md")  # noqa: F821
    REF_SMF = osp.join(RESULTS, "reference_smf_12panel.csv")  # noqa: F821
    SE_CSV = osp.join(RESULTS, "published_source_kuzmin_tau_se.csv")  # noqa: F821
    CALIB_CSV = osp.join(RESULTS, "published_source_pvalue_calibration.csv")  # noqa: F821
    PANEL_CSV = osp.join(RESULTS, "published_source_costanzo_panel.csv")  # noqa: F821
    INPANEL_CSV = osp.join(RESULTS, "published_source_inpanel_digenics.csv")  # noqa: F821

    # Verbatim source spans. Kept exactly as the OCR markdown stores them, including its
    # spaced-out LaTeX, so a re-OCR that changes them fails loudly instead of silently
    # invalidating a quote in the notes.
    Q_REPRO = (
        "The screen noise was similar for double mutants (Fig. S5B left) compared to raw "
        "triple mutant scores (Fig. S5B middle) with the correlation between independent "
        "replicates of 0.9-0.91. However, the adjusted trigenic interaction scores showed "
        "more variability with the correlation coefficient between replicates decreasing "
        "to 0.74-0.81 (Fig. S5B right)."
    )
    Q_CUTOFFS = (
        "the dashed lines represent intermediate, |score| $> 0 . 0 8$ , $p { < } 0 . 0 5$ "
        "(blue) and stringent, |score $> 0 . 1 2$ , $p { < } 0 . 0 5$ (red), cut-offs."
    )
    Q_VERY_STRINGENT = "and very stringent cut-offs $( \\tau < - 0 . 2 )$"
    Q_COSTANZO_THRESH = (
        "intermediate $( P <$ 0.05 and $\\left| \\varepsilon \\right| > 0 . 0 8 )$ , and "
        "stringent confidence ( $P < 0 . 0 5$ and $\\varepsilon > 0 . 1 6$ or "
        "$\\mathfrak { E } < - 0 . 1 2 )$"
    )

    def sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    def flat(path: str) -> str:
        """Note text with runs of whitespace collapsed, so a wrapped quote still matches."""
        return re.sub(r"\s+", " ", open(path).read())

    def table(path: str, key: str) -> dict:
        with open(path, newline="") as f:
            return {r[key]: r for r in csv.DictReader(f)}

    # -- 1. the two mirrored SI files are the ones the notes cite -------------------------
    k_sha, c_sha = sha256(KUZMIN_SI), sha256(COSTANZO_SI)
    check("published", k_sha == KUZMIN_SI_SHA,
          "Kuzmin 2018 si1.md sha256 matches the value quoted in the notes", k_sha[:16])
    check("published", c_sha == COSTANZO_SI_SHA,
          "Costanzo 2016 si1.md sha256 matches", c_sha[:16])

    kz = open(KUZMIN_SI).read()
    cz = open(COSTANZO_SI).read()

    # -- 2. reproducibility figures are verbatim, in the SI and in the note ---------------
    check("published", Q_REPRO in kz,
          "the r = 0.9-0.91 / 0.74-0.81 sentence is verbatim in the Kuzmin SI",
          "found in 'Evaluation of reproducibility of genetic interactions'")
    check("published", Q_REPRO in flat(HANDOFF),
          "run4-handoff quotes that sentence verbatim, whitespace aside", "quoted")
    check("published", "0.88" not in flat(HANDOFF).split("Kuzmin 2018 replicate")[1][:900]
          and "r = 0.59" not in flat(HANDOFF),
          "the retracted Fig. S5B read-offs (0.88, 0.59) are gone from the live text",
          "absent")

    # -- 3. the interaction-strength cut-offs --------------------------------------------
    check("published", Q_CUTOFFS in kz,
          "Kuzmin states intermediate |score| > 0.08 and stringent |score| > 0.12, and "
          "the caption covers the adjusted trigenic panel", "Fig. S5B caption")
    check("published", Q_VERY_STRINGENT in kz and "|\\tau| > 0 . 2" not in kz,
          "0.20 appears only as the NEGATIVE cut-off tau < -0.2, never as a magnitude",
          "Fig. S15 caption")
    rt_flat = flat(RATIONALE)  # noqa: F821
    check("published", "\\tau<-0.2" in rt_flat.replace(" ", "") or "tau<-0.2" in rt_flat,
          "the rationale note labels 0.20 as a negative cut-off, not a magnitude one",
          "labelled")
    check("published", Q_COSTANZO_THRESH in cz,
          "Costanzo states the intermediate threshold P<0.05 and |eps|>0.08 used for "
          "in-panel significance", "data-release section")

    # -- 4. Kuzmin's own SE(tau) ----------------------------------------------------------
    se = table(SE_CSV, "dataset")
    both = se["both"]
    check("published", float(both["p_max"]) == 0.5,
          "the released Kuzmin P-value tops out at exactly 0.5, so it is one-sided",
          both["p_max"])
    med1 = float(both["se_one_sided_median"])
    med2 = float(both["se_two_sided_median"])
    check("published", round(med1, 4) == 0.0785 and int(both["n_usable"]) == 392817,
          "median back-solved SE(tau) is 0.0785 over the invertible records",
          f"{med1:.4f} over n={both['n_usable']}")
    check("published", round(med2, 3) == 0.031,
          "the superseded 0.031 is exactly what a two-sided reading of the same p gives",
          f"{med2:.4f}")
    calib = table(CALIB_CSV, "reference_distribution")
    one, two = calib["one_sided_normal"], calib["two_sided_normal"]
    check("published",
          float(one["spearman_se_vs_sd"]) > float(two["spearman_se_vs_sd"])
          and float(one["relative_iqr"]) < float(two["relative_iqr"]),
          "one-sided beats two-sided on Kuzmin's own digenic SD column, so the "
          "one-sidedness is measured rather than assumed",
          f"spearman {one['spearman_se_vs_sd']} vs {two['spearman_se_vs_sd']}")
    check("published", "**0.0785**" in rt_flat and "is **0.031**" not in rt_flat,
          "the rationale note states 0.0785, not 0.031", "corrected")

    # -- 5. Costanzo coverage of the two zero-trigenic-data genes -------------------------
    panel = table(PANEL_CSV, "gene")
    check("published",
          int(panel["YER079W"]["smf_records"]) == 4
          and int(panel["YLR312C-B"]["smf_records"]) == 2,
          "YER079W has 4 smf_costanzo2016 records and YLR312C-B has 2",
          f"{panel['YER079W']['smf_records']} / {panel['YLR312C-B']['smf_records']}")
    check("published",
          int(panel["YER079W"]["smf_strains"]) == 2
          and int(panel["YLR312C-B"]["smf_strains"]) == 1,
          "those records are 2 distinct strains for YER079W and 1 for YLR312C-B, each "
          "stored twice", "2 / 1")
    check("published",
          int(panel["YLR312C-B"]["dmi_partners"]) == 3544
          and int(panel["YLR312C-B"]["dmi_significant_partners"]) == 99,
          "YLR312C-B has exactly 3,544 Costanzo digenic partners, 99 significant",
          f"{panel['YLR312C-B']['dmi_partners']} / "
          f"{panel['YLR312C-B']['dmi_significant_partners']}")
    check("published",
          int(panel["YER079W"]["dmi_partners"]) == 5232
          and int(panel["YER079W"]["dmi_significant_partners"]) == 386,
          "YER079W has 5,232 Costanzo digenic partners, 386 significant, so 3,544 belongs "
          "to YLR312C-B alone",
          f"{panel['YER079W']['dmi_partners']} / "
          f"{panel['YER079W']['dmi_significant_partners']}")
    check("published", "3,544" in rt_flat and "5,232" in rt_flat,
          "the rationale note attributes each partner count to its own gene", "both stated")

    # -- 6. the in-panel digenics of YLR312C-B --------------------------------------------
    inpanel = table(INPANEL_CSV, "pair")
    check("published", len(inpanel) == 55,
          "all 55 pairs of the eleven-gene basis are enumerated", len(inpanel))
    ylr = {k: v for k, v in inpanel.items() if "YLR312C-B" in k.split(" + ")}
    measured = sum(int(v["measured"]) for v in ylr.values())
    sig = sum(int(v["n_significant"]) for v in ylr.values())
    check("published", len(ylr) == 10 and measured == 10 and sig == 0,
          "YLR312C-B: 10 in-panel pairs, all 10 MEASURED in Costanzo, 0 significant, so "
          "this is measured and null and not unmeasured",
          f"{measured} measured, {sig} significant")
    check("published", max(float(v["max_abs_eps"]) for v in ylr.values()) < 0.08
          and min(float(v["min_p"]) for v in ylr.values()) > 0.05,
          "no YLR312C-B in-panel pair even reaches either half of the threshold",
          f"max |eps| {max(float(v['max_abs_eps']) for v in ylr.values()):.4f}, "
          f"min p {min(float(v['min_p']) for v in ylr.values()):.4f}")
    unmeasured = sorted(k for k, v in inpanel.items() if not int(v["measured"]))
    check("published",
          unmeasured == ["YLL012W + YLR104W", "YPL046C + YPL081W"],
          "only two in-panel pairs are unmeasured by Costanzo, and neither involves "
          "YLR312C-B", unmeasured)
    check("published", "0 of its 10 in-panel digenics" in flat(AUDIT_BRIEF),
          "the audit brief states 0 of 10, not 0 of 9 (which counted the TEN set)",
          "corrected")

    # -- 7. what R64-4-1 actually says about YLR312C-B ------------------------------------
    alias_lines, id_lines = [], set()
    with open(GFF) as f:
        for line in f:
            if "YLR312C-B" in line:
                alias_lines.append(line)
            for g in GENES:  # noqa: F821
                if f"ID={g};" in line:
                    id_lines.add(g)
    check("published",
          len(alias_lines) == 1 and "Alias=SPH1,YLR312C-B" in alias_lines[0]
          and "ID=YLR313C;" in alias_lines[0],
          "R64-4-1 carries YLR312C-B once, as an alias on the YLR313C (SPH1) gene line, "
          "with no feature of its own", len(alias_lines))
    check("published", id_lines == set(GENES) - {"YLR312C-B"},  # noqa: F821
          "every other panel gene is a first-class R64-4-1 feature",
          f"{len(id_lines)} of 11")

    # -- 8. the SMF reference panel now matches the run-4 plate ---------------------------
    ref = table(REF_SMF, "orf")
    blank = [g for g, r in ref.items() if not r["costanzo_smf"]]
    check("published", len(ref) == 12 and not blank,
          "the SMF reference panel covers all 12 run-4 strains with a published Costanzo "
          "value", f"{len(ref)} rows, {len(blank)} blank")
    check("published", "YLR104W" in ref and "YPL056C" not in ref,
          "the panel carries YLR104W (LCL2, the strain on the run-4 plate) and not "
          "YPL056C (LCL1, the earlier design)", "LCL2 in, LCL1 out")
    check("published",
          float(ref["YLR104W"]["costanzo_smf"]) == float(panel["YLR104W"]["smf_fitness"])
          and float(ref["YLR104W"]["costanzo_se"]) == float(panel["YLR104W"]["smf_std"]),
          "the panel's YLR104W value equals smf_costanzo2016 read from the LMDB, so it is "
          "sourced rather than transcribed",
          f"{ref['YLR104W']['costanzo_smf']} +/- {ref['YLR104W']['costanzo_se']}")
    with open(MEAS_GAPS, newline="") as f:  # noqa: F821
        gaps = list(csv.DictReader(f))
    check("published",
          len(gaps) == 3
          and not [g for g in gaps if g["kind"] == "single, no published reference"],
          "no run-4 single is recorded as lacking a published SMF, and 3 gaps remain",
          [g["kind"] for g in gaps])
    s7 = [ln for ln in open(BUILD_LIST) if ln.startswith("| s7 ")]  # noqa: F821
    check("published",
          len(s7) == 1 and s7[0].strip().strip("|").split("|")[-1].strip() == "1.0322",
          "the build-list s7 row carries the published SMF instead of --",
          s7[0].strip() if s7 else "no s7 row")

    # -- 9. the digenic that sits inside selected triple rank 5 ---------------------------
    row = inpanel["YER079W + YLR104W"]
    strongest = max(inpanel.values(), key=lambda v: float(v["max_abs_eps"] or 0))
    check("published",
          int(row["n_significant"]) == 1 and float(row["max_abs_eps"]) == 0.5672
          and float(row["min_p"]) == 0.000183 and strongest is row,
          "YER079W x YLR104W is significant and is the largest-magnitude in-panel digenic "
          "of all 53 measured pairs", f"eps -0.5672, P {row['min_p']}")
    check("published",
          "-0.5672" in rt_flat and "1.832" in rt_flat and "-0.0217" in rt_flat,
          "the rationale note records that digenic, its P, and the reciprocal screen that "
          "does not reproduce it", "recorded")


# ----------------------------------------------------------------------------------------
# standalone harness; delete when splicing into verify_triple_build_list.py
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.environ["DATA_ROOT"]
    _S = ("/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/"
          "76d2d692-76e5-4bb8-8500-bc00a868cb2d/scratchpad/audit")
    # scratch stand-ins: results/ holds published_source_audit.py's output, notes_corrected/
    # holds the two notes with this audit's edits applied
    RESULTS = osp.join(_S, "results")
    NOTES = osp.join(_S, "notes_corrected")
    RATIONALE = osp.join(NOTES, "experiments.W019-echo-crispr-array.next-strains-to-construct.md")
    BUILD_LIST = osp.join(NOTES, "experiments.W019-echo-crispr-array.build-list.md")
    MEAS_GAPS = osp.join(RESULTS, "run4_measured_summary_gaps.csv")
    GENES = frozenset((
        "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
        "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
    ))
    NO_TRIGENIC_DATA = frozenset(("YER079W", "YLR312C-B"))

    _tally = []

    def _check(group: str, ok: bool, claim: str, observed: object) -> None:
        _tally.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {claim}  [{observed}]")

    run(_check)
    n_fail = _tally.count(False)
    print(f"\n{len(_tally)} checks, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
