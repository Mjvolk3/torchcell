#!/usr/bin/env python
"""Count MAGIC guide barcodes per SRA run.

Barcode = read[27:70] (43 bp, activation) or read[27:71] (44 bp, interference/
deletion), forward orientation -- determined empirically (offset-27 scan; ~79%
exact-match for activation/furfural, ~63% for deletion, consistent within a
library so the after/before enrichment RATIO is unbiased). Exact match against the
100,493-guide reference (Supplementary Data 4 / MOESM6). Writes:
  counts/<run>.tsv      guide_id<TAB>count   (mapped guides only)
  counts/qc.tsv         run, total_reads, mapped_reads, mapping_rate
"""

import csv
import gzip
import os
from collections import Counter

import pandas as pd

WORK = os.environ["LIAN_WORK"]
REF = os.path.join(os.environ["LIAN_INPUTS"], "41467_2019_13621_MOESM6_ESM.xlsx")
OFF = 27

ref = pd.read_excel(REF, header=None)
ref.columns = ["id", "seq"]
ref["seq"] = ref["seq"].astype(str).str.strip()
seq2id = {s: i for i, s in zip(ref["id"], ref["seq"])}
print(f"reference guides: {len(seq2id)}", flush=True)

manifest = pd.read_csv(os.path.join(WORK, "run_manifest.tsv"), sep="\t")
os.makedirs(os.path.join(WORK, "counts"), exist_ok=True)
qc_rows = []

for run in manifest["run"]:
    fq = os.path.join(WORK, "sra", f"{run}.fastq.gz")
    counts = Counter()
    total = 0
    mapped = 0
    with gzip.open(fq, "rt") as fh:
        for i, line in enumerate(fh):
            if i & 3 != 1:
                continue
            total += 1
            r = line
            gid = seq2id.get(r[OFF : OFF + 43]) or seq2id.get(r[OFF : OFF + 44])
            if gid is not None:
                counts[gid] += 1
                mapped += 1
    out = os.path.join(WORK, "counts", f"{run}.tsv")
    with open(out, "w", newline="") as o:
        w = csv.writer(o, delimiter="\t")
        for gid, c in counts.items():
            w.writerow([gid, c])
    rate = 100 * mapped / total if total else 0.0
    qc_rows.append([run, total, mapped, round(rate, 2)])
    print(
        f"{run}: total={total} mapped={mapped} ({rate:.1f}%) guides={len(counts)}",
        flush=True,
    )

with open(os.path.join(WORK, "counts", "qc.tsv"), "w", newline="") as o:
    w = csv.writer(o, delimiter="\t")
    w.writerow(["run", "total_reads", "mapped_reads", "mapping_rate_pct"])
    w.writerows(qc_rows)
print("COUNT_COMPLETE", flush=True)
