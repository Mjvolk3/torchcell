---
id: eie9u8lczb1yf8ukgty4pd6
title: verify_triple_build_list
desc: ''
updated: 1787527303840
created: 1787527303840
---

## 2026.08.23 - Independent audit of the capped triple build list

`triple_design_rank_sampling.py` chooses the round; this script checks it. It does not
import the design script -- it re-derives the strain inventory, the target basis and the
selection's consequences from the pinned inputs, then asserts every number the two bench
notes state. **52 checks, 0 fail** as of 2026.08.23.

Run from repo root:

```bash
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py
```

Exit code is 1 if any check fails, so it can gate a change to the design script or to
either note.

### What it checks

| group | checks | covers |
|---|---|---|
| inventory | 8 | 12 singles / 13 doubles / 0 triples, YLR313C is the one non-node single, YLR104W has no built doubles, blocked pair absent |
| basis | 7 | 39 in-basis targets, ranks 1-10 all zero-data, orphan targets are 26/32/37, set-cover 31 of 31 intact |
| selection | 5 | 20 distinct in-basis triples, cap of 6 held, every triple has a built parent, 15 single-route |
| doubles | 5 | 25 new, 8 existing re-measured, 33 distinct on plate, all parents built |
| tau | 5 | **the load-bearing one** -- see below |
| contingency | 5 | 14 triples and 17 needed doubles survive dropping the two zero-data genes |
| notes | 12 | every row of both note tables against the computed selection |
| kuzmin | 3 | per-gene record counts read straight from the LMDBs |

### The load-bearing check

A trigenic interaction consumes seven measured fitnesses:

$$\tau_{abc} = f_{abc} - f_{ab}f_c - f_{ac}f_b - f_{bc}f_a + 2 f_a f_b f_c$$

so a triple yields nothing unless all three of its doubles and all three of its singles
are on the same plate. A design that picked 20 triples without closing over their doubles
would produce 20 fitness numbers and zero tau values. Group `tau` asserts that after the
45 strains are built, all 20 triples have all six supporting terms, and that the 65 plate
strains are exactly the closure of that requirement -- nothing extra, nothing missing.

### Kuzmin coverage is read, not quoted

Group `kuzmin` opens `tmi_kuzmin{2018,2020}` (91,111 + 301,798 records, every one a 3-KO)
and `dmi_kuzmin{2018,2020}` (410,399 + 632,797, every one a 2-KO) and counts records per
panel gene. It confirms YER079W and YLR312C-B are the only two genes absent, from both,
and that the trigenic column in
[[experiments.W019-echo-crispr-array.next-strains-to-construct]] matches the LMDBs. Only
the trigenic datasets are in the training query
(`experiments/010-kuzmin-tmi/queries/001_small_build.cql`); the digenic counts are context.
This pass reads ~1.4M records and takes a couple of minutes.

### Outputs

- `experiments/W019-echo-crispr-array/results/verify_triple_build_list_checks.csv` -- one
  row per assertion (group, pass, claim, observed)
- `experiments/W019-echo-crispr-array/results/verify_triple_build_list_kuzmin.csv` --
  per-gene trigenic + digenic counts

### What it found

One error, in the drop-the-two contingency. Both notes said 18 new doubles survive if
YER079W and YLR312C-B are dropped. 18 of the 25 contain neither gene, but
`YLR104W + YPL081W` serves only rank 3 (`YLR104W + YLR312C-B + YPL081W`), so 17 feed a
surviving triple and the core is 31 strains, not 32 or the "34" the rationale note also
claimed. Both notes corrected.

### Related

- [[experiments.W019-echo-crispr-array.build-list]]
- [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]]
