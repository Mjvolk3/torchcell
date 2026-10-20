---
id: ykm1g8b4371kkaps7zonrcq
title: Count_distinct_alleles
desc: ''
updated: 1792474341627
created: 1792474341627
---

## 2026.10.20 - Distinct alleles of the Peter 2018 gene-keyed tarball

One streaming CPU pass over
`$DATA_ROOT/torchcell-genomes/peter2018_1011_assemblies/allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`
(about 171 s on GilaHyper). For every gene file it hashes each isolate's CDS (sha1 of the
upper-cased sequence), counts distinct CDS, translates the distinct CDS truncated to whole
codons (Biopython standard table) and counts distinct proteins. Writes
`experiments/043-expression-proteome-morphology/results/distinct_alleles.json`: totals,
per-gene counts and the sorted isolate ids seen in the tarball (so a segregant parent's
Peter id can be checked for membership).

```bash
python experiments/043-expression-proteome-morphology/scripts/count_distinct_alleles.py
```

Result (run 2026-10-20, identical per-gene counts to the review pass's first run):

- 6,015 gene files, 1,011 isolates, 6,081,165 allele records, mean CDS 1,438 bp.
- 1,198,539 distinct CDS alleles (19.7% of records), 775,856 distinct protein alleles
  (12.8%); per gene median 188 CDS and 109 proteins, maximum 991 and 967.
- The five non-BY segregant parents (RM `AAA`, YJM145 `AAS`, YPS163 `ACD`, CBS2888
  `ABL`, YJM981 `ADI`) are all among the 1,011 isolate ids.

Read by [[experiments.043-expression-proteome-morphology.scripts.primer_043_tables]] for
the data-per-part table (`t14-data-per-part.tex`) and the I4, D1 and E1 decision rows of
the 043 datasets primer. The distinct protein count sets the isolate embedding bill
(about 2.7 h of ProtT5 on two cards under stated throughput assumptions; untested).
