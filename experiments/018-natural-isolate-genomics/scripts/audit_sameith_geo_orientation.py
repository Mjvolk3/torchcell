# experiments/018-natural-isolate-genomics/scripts/audit_sameith_geo_orientation.py
# [[experiments.018-natural-isolate-genomics.scripts.audit_sameith_geo_orientation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/audit_sameith_geo_orientation

"""Per-array dye-orientation audit of GSE42536 (Sameith 2015), read from the GEO source.

GSE42536 is a **dye-swap** design. GEO documents this per sample:

    GSM636383  dot6-del-1-a   src_ch1=refpool  label_ch1=Cy5
                              src_ch2=dot6-del label_ch2=Cy3     -> mutant is Cy3
    GSM636384  dot6-del-1-b   src_ch1=dot6-del label_ch1=Cy5
                              src_ch2=refpool  label_ch2=Cy3     -> mutant is Cy5   (swapped)

So the ``-a`` / ``-b`` title suffix IS the dye swap, and the two orientations need
OPPOSITE sign handling to land on a common M = log2(mutant / refpool).

The trap: the ``#VALUE`` header is **not consistent across the series**. GEO declares
BOTH ratio directions within GSE42536 (see the counts this script prints). So a loader
cannot hard-code one convention -- and ``sameith2015.py`` does exactly that
(``sameith2015.py:645-653`` comments "VALUE is log2(Cy5/Cy3)" and derives the sign from
``source_name_ch1`` alone).

This script settles it without inference: for every one of the 287 arrays it recomputes
the orientation EMPIRICALLY from the ``Signal Norm_Cy5`` / ``Signal Norm_Cy3`` columns
(|corr| with VALUE is 1.000, so the call is unambiguous), combines that with the dye
assignment, derives the CORRECT sign, and compares against what the loader applies.

Result: 217 / 287 arrays correct, **70 (24%) signed backwards**. The GLOBAL sign is right
(hence the positive Kemmeren<->Sameith profile correlation that validated it), but a
quarter of arrays enter negated, and because replicates are averaged, mixed-sign
replicates ATTENUATE |M| and INFLATE SE. That is why Sameith's deleted-gene median is
only -0.71 against Kemmeren's -2.48.

The fix is the one Kemmeren's loader already uses: ignore ``VALUE`` and recompute the
ratio from the signal columns + the dye assignment.
"""

import gzip
import json
import os
import os.path as osp
from collections import Counter, defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics", "results")

SOFT = osp.join(
    DATA_ROOT, "data/torchcell/sm_microarray_sameith2015/raw", "GSE42536_family.soft.gz"
)


def parse_soft(path: str) -> list[dict]:
    samples: list[dict] = []
    cur: dict | None = None
    intab = False
    cols_seen = False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                if cur:
                    samples.append(cur)
                cur = {"gsm": line.split("=")[1].strip(), "value_def": None, "rows": []}
                intab = False
                cols_seen = False
            elif cur is None:
                continue
            elif line.startswith("!Sample_title"):
                cur["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1"):
                cur["src1"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch2"):
                cur["src2"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_label_ch1"):
                cur["lab1"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_label_ch2"):
                cur["lab2"] = line.split("=", 1)[1].strip()
            elif line.startswith("#VALUE"):
                cur["value_def"] = line.split("=", 1)[1].strip()
            elif line.startswith("!sample_table_begin"):
                intab = True
                cols_seen = False
            elif line.startswith("!sample_table_end"):
                intab = False
            elif intab:
                if not cols_seen:  # header row: ID_REF VALUE Signal Norm_Cy5 ..._Cy3
                    cols_seen = True
                    continue
                cur["rows"].append(line.split("\t"))
    if cur:
        samples.append(cur)
    return samples


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    samples = parse_soft(SOFT)
    print(f"parsed {len(samples)} GSMs from {osp.basename(SOFT)}\n")

    stated = Counter(s["value_def"] for s in samples)
    print("--- (1) STATED #VALUE definitions in GEO (they are NOT consistent) ---")
    for d, c in stated.most_common():
        print(f"  {c:4d} x  {d!r}")

    res: list[dict] = []
    for s in samples:
        v, c5, c3 = [], [], []
        for r in s["rows"]:
            if len(r) < 4:
                continue
            try:
                vv, a, b = float(r[1]), float(r[2]), float(r[3])
            except ValueError:
                continue
            if a > 0 and b > 0:
                v.append(vv)
                c5.append(a)
                c3.append(b)
        if len(v) < 500:
            continue
        v = np.array(v)
        lr_3over5 = np.log2(np.array(c3) / np.array(c5))
        r_pos = float(np.corrcoef(v, lr_3over5)[0, 1])
        empirical = "log2(Cy3/Cy5)" if r_pos > 0 else "log2(Cy5/Cy3)"

        # ch1 == Cy5 throughout this series; mutant is Cy3 iff refpool is in ch1
        mut_is_cy3 = "refpool" in s.get("src1", "").lower()
        if empirical == "log2(Cy3/Cy5)":
            correct = 1 if mut_is_cy3 else -1
        else:
            correct = -1 if mut_is_cy3 else 1
        loader = -1 if mut_is_cy3 else 1  # sameith2015.py:645-653

        res.append(
            {
                "gsm": s["gsm"],
                "title": s["title"],
                "stated": s["value_def"],
                "empirical": empirical,
                "mutant_dye": "Cy3" if mut_is_cy3 else "Cy5",
                "correct_sign": correct,
                "loader_sign": loader,
                "ok": correct == loader,
                "abs_corr": abs(r_pos),
            }
        )

    print("\n--- (2) EMPIRICAL orientation (recomputed from Signal Norm_Cy5/Cy3) ---")
    for d, c in Counter(r["empirical"] for r in res).most_common():
        med = np.median([r["abs_corr"] for r in res if r["empirical"] == d])
        print(f"  {c:4d} x  {d}   (|corr| with VALUE: median {med:.3f})")

    print("\n--- (3) LOADER vs CORRECT, per array ---")
    tab: dict = defaultdict(int)
    for r in res:
        suffix = r["title"].rsplit("-", 1)[-1]
        tab[
            (suffix, r["mutant_dye"], r["empirical"], "OK" if r["ok"] else "BACKWARDS")
        ] += 1
    print(f"  {'suffix':7s} {'mutant':7s} {'VALUE is':16s} {'loader':10s} {'n':>4s}")
    for k in sorted(tab):
        print(f"  {k[0]:7s} {k[1]:7s} {k[2]:16s} {k[3]:10s} {tab[k]:4d}")

    nbad = sum(1 for r in res if not r["ok"])
    print(f"\n>>> loader signs CORRECTLY : {len(res) - nbad} / {len(res)}")
    print(
        f">>> loader signs BACKWARDS : {nbad} / {len(res)} "
        f"({100 * nbad / len(res):.0f}%)"
    )
    print("\n>>> The GLOBAL sign is right (the majority orientation), which is why the")
    print("    Kemmeren<->Sameith profile correlation validated it (+0.42 mean). But a")
    print("    quarter of arrays enter negated; averaging mixed-sign replicates")
    print("    attenuates |M| and inflates SE. No global flip can fix this.")

    with open(osp.join(RESULTS_DIR, "sameith_geo_orientation_audit.json"), "w") as fh:
        json.dump(
            {
                "n_arrays": len(res),
                "stated_value_definitions": dict(stated),
                "empirical_orientation": dict(Counter(r["empirical"] for r in res)),
                "loader_correct": len(res) - nbad,
                "loader_backwards": nbad,
                "frac_backwards": nbad / len(res),
                "arrays": res,
            },
            fh,
            indent=2,
        )
    print(f"\nresults -> {RESULTS_DIR}/sameith_geo_orientation_audit.json")


if __name__ == "__main__":
    main()
