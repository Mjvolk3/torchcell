---
id: 7qm2vd0jx4nb8afkw3rc1ye
title: build-list
desc: ''
updated: 1786931101296
created: 1786931101296
---

## 2026.08.23 - Strains to construct: 25 doubles + 20 triples

**45 new strains.** All are built from strains that already exist in the run-4 panel -- no
new single knockouts are needed, and **every one builds in parallel**: nothing waits on
anything else.

**The design on this sheet is `capped`.** Six selection strategies were compared over the
same 39 candidate triples; `capped` is the one being built, and every ID below comes from
it. It takes the six best-predicted triples that involve a gene the model has no trigenic
training data for, then fills the remaining fourteen from further down the ranking, which
holds those genes to six of twenty. The comparison is under
"[Where this list came from](#where-this-list-came-from)"; the full argument is in
[[experiments.W019-echo-crispr-array.next-strains-to-construct]].

Two conventions used throughout:

- **`*` on an ID** means the strain involves `YER079W` or `YLR312C-B`, the two genes with
  zero trigenic training records. Those predictions are extrapolation.
- **`routes`** is how many already-built doubles a triple can be made from. `1` means one
  parent only, so a failed cross has no backup until the matching new double is done.

Ordering suggestion: **T01--T15 first**, the fifteen at `routes` 1. T16--T20 each have a
second route.

### What already exists

From the run-4 order sheet, **12 singles, 13 doubles, 0 triples**:

```text
experiments/W019-echo-crispr-array/data/run4_doubles_2026-08-06/
  Single-and-Double-KO-Strains-List-Order.csv
```

**Singles s1--s12.** Eleven are prediction nodes and all eleven are used below. `s11`
YLR313C (SPH1) was built but was never a node in the inference panel, so it is in no
predicted triple and is not on this plate.

| id  | ORF        | common   | id  | ORF        | common                  |
|:----|:-----------|:---------|:----|:-----------|:------------------------|
| s1  | YJR060W    | CBF1     | s7  | YLR104W    | LCL2                    |
| s2  | YPL081W    | RPS9A    | s8  | YLL012W    | YEH1                    |
| s3  | YDR057W    | YOS9     | s9  | YER079W    | --                      |
| s4  | YPL046C    | ELC1     | s10 | YKL033W-A  | --                      |
| s5  | YBR203W    | COS111   | s11 | YLR313C    | SPH1 (not a panel node) |
| s6  | YGL087C    | MMS2     | s12 | YLR312C-B  | --                      |

**Doubles d1--d13.** Eight are parents for the triples below and must be re-measured on
this plate; five are not used by any selected triple.

| id | pair | used | serves triples (rank) |
|---|---|---|---|
| d1 | YDR057W + YPL081W | yes | 12, 13, 16 |
| d2 | YPL046C + YPL081W | no | -- |
| d3 | YER079W + YPL081W | no | -- |
| d4 | YLR312C-B + YPL081W | yes | 1, 3, 4, 6 |
| d5 | YDR057W + YGL087C | yes | 16, 39 |
| d6 | YDR057W + YER079W | no | -- |
| d7 | YDR057W + YLR312C-B | no | -- |
| d8 | YJR060W + YPL046C | yes | 21, 31, 33 |
| d9 | YBR203W + YPL046C | yes | 24, 28, 29, 31, 34 |
| d10 | YDR057W + YLL012W | yes | 25, 38, 39 |
| d11 | YLL012W + YPL046C | yes | 21, 24, 27 |
| d12 | YER079W + YJR060W | no | -- |
| d13 | YER079W + YLR312C-B | yes | 2, 5 |

The 14th designed double, `YKL033W-A x YJR060W`, failed to construct and does not exist.

### Part 1 -- 25 double knockouts

Cross the two singles listed. Both parents already exist.

| ID   | gene 1           | gene 2           | ID   | gene 1           | gene 2           |
|:-----|:-----------------|:-----------------|:-----|:-----------------|:-----------------|
| D01  | YBR203W (COS111) | YDR057W (YOS9)   | D14  | YGL087C (MMS2)   | YPL046C (ELC1)   |
| D02  | YBR203W (COS111) | YGL087C (MMS2)   | D15  | YGL087C (MMS2)   | YPL081W (RPS9A)  |
| D03  | YBR203W (COS111) | YJR060W (CBF1)   | D16  | YJR060W (CBF1)   | YLL012W (YEH1)   |
| D04  | YBR203W (COS111) | YKL033W-A        | D17  | YJR060W (CBF1)   | YLR104W (LCL2)   |
| D05  | YBR203W (COS111) | YLL012W (YEH1)   | D18  | YKL033W-A        | YLL012W (YEH1)   |
| D06  | YBR203W (COS111) | YLR104W (LCL2)   | D19* | YKL033W-A        | YLR312C-B        |
| D07* | YBR203W (COS111) | YLR312C-B        | D20  | YKL033W-A        | YPL046C (ELC1)   |
| D08  | YBR203W (COS111) | YPL081W (RPS9A)  | D21  | YKL033W-A        | YPL081W (RPS9A)  |
| D09  | YDR057W (YOS9)   | YKL033W-A        | D22* | YLL012W (YEH1)   | YLR312C-B        |
| D10* | YER079W          | YLL012W (YEH1)   | D23* | YLR104W (LCL2)   | YLR312C-B        |
| D11* | YER079W          | YLR104W (LCL2)   | D24  | YLR104W (LCL2)   | YPL046C (ELC1)   |
| D12  | YGL087C (MMS2)   | YLL012W (YEH1)   | D25  | YLR104W (LCL2)   | YPL081W (RPS9A)  |
| D13* | YGL087C (MMS2)   | YLR312C-B        |      |                  |                  |

### Part 2 -- 20 triple knockouts

`build from` is an **existing** double from run 4; cross it with the single in
`x third single`. `routes` counts the already-built doubles this triple could be made
from; start with the fifteen at `1`.

| ID   | gene 1           | gene 2           | gene 3           | build from                | x third single   | routes |
|:-----|:-----------------|:-----------------|:-----------------|:--------------------------|:-----------------|:-------|
| T01* | YKL033W-A        | YLR312C-B        | YPL081W (RPS9A)  | YLR312C-B + YPL081W       | YKL033W-A        | 1      |
| T02* | YER079W          | YLL012W (YEH1)   | YLR312C-B        | YER079W + YLR312C-B       | YLL012W (YEH1)   | 1      |
| T03* | YLR104W (LCL2)   | YLR312C-B        | YPL081W (RPS9A)  | YLR312C-B + YPL081W       | YLR104W (LCL2)   | 1      |
| T04* | YBR203W (COS111) | YLR312C-B        | YPL081W (RPS9A)  | YLR312C-B + YPL081W       | YBR203W (COS111) | 1      |
| T05* | YER079W          | YLR104W (LCL2)   | YLR312C-B        | YER079W + YLR312C-B       | YLR104W (LCL2)   | 1      |
| T06* | YGL087C (MMS2)   | YLR312C-B        | YPL081W (RPS9A)  | YLR312C-B + YPL081W       | YGL087C (MMS2)   | 1      |
| T07  | YBR203W (COS111) | YDR057W (YOS9)   | YPL081W (RPS9A)  | YDR057W + YPL081W         | YBR203W (COS111) | 1      |
| T08  | YDR057W (YOS9)   | YKL033W-A        | YPL081W (RPS9A)  | YDR057W + YPL081W         | YKL033W-A        | 1      |
| T09  | YDR057W (YOS9)   | YKL033W-A        | YLL012W (YEH1)   | YDR057W + YLL012W         | YKL033W-A        | 1      |
| T10  | YGL087C (MMS2)   | YLL012W (YEH1)   | YPL046C (ELC1)   | YLL012W + YPL046C         | YGL087C (MMS2)   | 1      |
| T11  | YBR203W (COS111) | YLR104W (LCL2)   | YPL046C (ELC1)   | YBR203W + YPL046C         | YLR104W (LCL2)   | 1      |
| T12  | YBR203W (COS111) | YKL033W-A        | YPL046C (ELC1)   | YBR203W + YPL046C         | YKL033W-A        | 1      |
| T13  | YJR060W (CBF1)   | YLR104W (LCL2)   | YPL046C (ELC1)   | YJR060W + YPL046C         | YLR104W (LCL2)   | 1      |
| T14  | YBR203W (COS111) | YGL087C (MMS2)   | YPL046C (ELC1)   | YBR203W + YPL046C         | YGL087C (MMS2)   | 1      |
| T15  | YBR203W (COS111) | YDR057W (YOS9)   | YLL012W (YEH1)   | YDR057W + YLL012W         | YBR203W (COS111) | 1      |
| T16  | YDR057W (YOS9)   | YGL087C (MMS2)   | YPL081W (RPS9A)  | YDR057W + YGL087C         | YPL081W (RPS9A)  | 2      |
| T17  | YJR060W (CBF1)   | YLL012W (YEH1)   | YPL046C (ELC1)   | YJR060W + YPL046C         | YLL012W (YEH1)   | 2      |
| T18  | YBR203W (COS111) | YLL012W (YEH1)   | YPL046C (ELC1)   | YBR203W + YPL046C         | YLL012W (YEH1)   | 2      |
| T19  | YBR203W (COS111) | YJR060W (CBF1)   | YPL046C (ELC1)   | YBR203W + YPL046C         | YJR060W (CBF1)   | 2      |
| T20  | YDR057W (YOS9)   | YGL087C (MMS2)   | YLL012W (YEH1)   | YDR057W + YGL087C         | YLL012W (YEH1)   | 2      |

### \* on an ID = involves a gene absent from our trigenic library

`YER079W` and `YLR312C-B` have **zero** trigenic records in the datasets the prediction
model was trained on, so predictions involving them are extrapolation. Exposure is
deliberately limited to **6 of 20 triples** (T01, T02, T03, T04, T05, T06) and **7 of 25
doubles** (D07, D10, D11, D13, D19, D22, D23).

If those two genes are later dropped, **14 triples survive and 17 of the 25 new doubles are
still needed** (D25, `YLR104W + YPL081W`, would then feed no triple). Worth settling before
starting the starred strains.

### Where this list came from

There were 39 candidate triples. The figure shows which of them each of the six strategies
would pick, strongest-predicted at the top. **Read the `capped` column: that is this
sheet.** It takes the six best-predicted triples that involve a starred gene, then fills
the rest from further down the list to hold the starred genes to six. `rank` would have
put 17 of 20 on those genes and `balanced` 15; `capped` is the only design that holds the
number to a chosen value while still covering all eleven genes.

![Which of the 39 candidate triples each design selects](assets/images/W019-echo-crispr-array/triple_design_rank_sampling.svg)

How often each gene ends up in a triple, again reading the `capped` bars. It spreads
across all eleven genes, range 2 to 8, rather than concentrating on a few.

![How often each gene appears](assets/images/W019-echo-crispr-array/triple_design_gene_frequency.svg)

### Notes for the bench

- **Eleven singles are used** and all already exist: YBR203W (COS111), YDR057W (YOS9),
  YER079W, YGL087C (MMS2), YJR060W (CBF1), YKL033W-A, YLL012W (YEH1), YLR104W (LCL2),
  YLR312C-B, YPL046C (ELC1), YPL081W (RPS9A).
- **Do not attempt `YKL033W-A x YJR060W`.** It failed to construct previously and is
  deliberately not in this list. If anyone retries it, please record the attempt date and
  outcome -- we have no record of when the original attempt happened.
- **`YLR104W` (LCL2) has no doubles in the current panel.** Six of the new doubles pair
  with it (D06, D11, D13, D17, D24, D25); those are the first coverage this gene has had.
- **`YLR312C-B`** is a merged/deprecated ORF; there is an open question about whether to
  use it or the adjacent real gene SPH1 (YLR313C). It appears in D07, D13, D19, D22, D23
  and among the starred triples.
- The plating round also needs the existing singles plus **8 of the 13
  existing doubles** as parents.

### Plate

| | count |
|---|---|
| singles | 11 |
| doubles | 33 (8 existing + 25 new) |
| triples | 20 |
| WT (BY4741) | 1 |
| **on plate** | **65** |
| **to construct** | **45** |

5 wells per strain on one 384 layout; 65 tubes at one pick per strain.

**Why all 65 and not just the 20 triples.** A trigenic interaction consumes seven measured
fitnesses:

$$\tau_{abc} = f_{abc} - f_{ab}f_c - f_{ac}f_b - f_{bc}f_a + 2 f_a f_b f_c$$

so every triple needs its own fitness plus **all three** of its doubles and **all three** of
its singles on the same plate. Verified against the selection: after the 45 strains are
built, all 20 triples have all six supporting terms present, and the 65 strains listed here
are exactly the closure of that requirement -- no extra, none missing. What is bought is
20 computable tau values, not 20 fitness numbers.

Full rationale and measurement caveats:
[[experiments.W019-echo-crispr-array.next-strains-to-construct]]
