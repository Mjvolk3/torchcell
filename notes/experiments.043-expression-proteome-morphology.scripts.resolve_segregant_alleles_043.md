---
id: gfdpn5t1bqrdih66th3ln5j
title: Resolve_segregant_alleles_043
desc: ''
updated: 1792638781908
created: 1792638781908
---

## 2026.10.21 - Every segregant record resolved to parent alleles

Script: `experiments/043-expression-proteome-morphology/scripts/resolve_segregant_alleles_043.py`; launcher [[experiments.043-expression-proteome-morphology.scripts.gh_resolve_segregant_alleles_043]]; rule in [[torchcell.data.representation_043]].

Slurm job 6118 (commit `12acf06be`, 32 workers): 82,467 records (Boocock 79,516, N'Guessan 2,951) in 3 min 7 s. Output: `$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/representation_043/segregant_alleles.parquet` (one row per record keyed by `raw_index`, with the 6,607-byte assignment and reason vectors) and `results/resolve_segregant_alleles_043.json`. Every record's mark count equals its non-reference assignment count (checked per record in the run).

Genes per record under the definite rule (posterior >= 0.9, whole gene in one confident run), median (mean); 6,607 genes:

| Dataset : cross | Parents (Peter id) | Records | Assigned | Assigned BY | Non-reference (marked) | Non-ref allele differs from S288C | Non-ref without allele | Unassigned: no block | Low posterior | Straddle | Assigned, single-block rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Boocock : A | BY, RM (AAA) | 28,137 | 4,431 (4,448) | 2,278 (2,291) | 2,136 (2,157) | 1,801 (1,820) | 176 (177) | 1,317 (1,295) | 734 (779) | 85 (85) | 3,888 (3,931) |
| Boocock : B | YJM145 (AAS), YPS163 (ACD) | 44,784 | 3,814 (3,887) | 0 | 3,814 (3,887) | 3,291 (3,351) | 314 (320) | 1,529 (1,506) | 1,128 (1,131) | 79 (82) | 3,177 (3,291) |
| Boocock : 3004 | CBS2888 (ABL), YJM981 (ADI) | 6,595 | 4,376 (4,049) | 0 | 4,376 (4,049) | 3,776 (3,492) | 362 (337) | 1,584 (1,584) | 569 (911) | 63 (64) | 3,473 (3,338) |
| N'Guessan : BBQTL_BYxRM | BY4741, RM11-1a (AAA) | 2,951 | 6,022 (5,762) | 2,896 (2,886) | 2,910 (2,876) | 2,472 (2,442) | 230 (229) | 380 (412) | 155 (380) | 52 (53) | 5,889 (5,595) |

Reading the table:

- "Non-ref allele differs from S288C" counts marked genes whose four keys are all present and at least one differs from the S288C key (windows included); "without allele" counts marked genes outside the parent's released allele set (the 596 genes of [[experiments.043-expression-proteome-morphology.scripts.representation_043]]).
- "No block" is a gene that overlaps no block at all: chromosome ends before the first or after the last marker, the mitochondrial genes, and genes lying wholly in the marker gap at a parent switch or next to a low-posterior block. It is the largest unassigned class on Boocock (median 1,317 to 1,584 genes per cell), where the markers are LD-pruned and the de novo posteriors dip often.
- The literal single-block rule assigns 543 fewer genes per Boocock A cell (median 3,888 against 4,431) and 133 fewer per N'Guessan record (5,889 against 6,022): those genes span a posterior-level split inside one parent's run.

Distinct parent alleles the records use (genes ever assigned to that parent in any record; keys from the isolate index):

| Parent | Genes ever assigned | With all four keys | Distinct protein keys | Distinct CaLM keys | Allele differs from S288C |
|---|---:|---:|---:|---:|---:|
| AAA (RM11-1a) | 6,446 | 5,689 | 5,879 | 5,905 | 5,394 |
| AAS (YJM145) | 6,439 | 5,639 | 5,874 | 5,896 | 5,366 |
| ABL (CBS2888) | 6,419 | 5,626 | 5,862 | 5,885 | 5,623 |
| ACD (YPS163) | 6,439 | 5,694 | 5,870 | 5,896 | 5,685 |
| ADI (YJM981) | 6,419 | 5,696 | 5,861 | 5,884 | 5,410 |

`sacct` MaxRSS for 6118 reads 107 GB against a 64 GB request; it sums the 32 forked workers' resident pages, which share the context tables copy-on-write, and the cgroup did not kill the job, so the true peak is below 64 GB (not measured separately).
