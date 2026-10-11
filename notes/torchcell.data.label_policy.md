---
id: l1s0od2g99eukfbtkymwgi8
title: Label_policy
desc: ''
updated: 1789963193485
created: 1789963193485
---

## 2026.09.21 - Why the choice moved out of the build

The 025 build chose at build time: `MeanExperimentDeduplicator` averaged every entry sharing an
experiment type and gene set into one value and replaced the source p-values with a t-test over
the merged scores. That one operation produced three of the five hazards the S3 closure recompute
measured, and undoing it cost a second 44 hour build. The 029 build drops the stage and keeps every
entry, which is why something has to choose at read time.

`LabelPolicy` is that chooser: a frozen pydantic object, hashed to a 16 character `policy_id`, that
takes one record's entries and returns one value per label plus the entry that supplied it.
Changing a rule costs a new cached table, not a rebuild, and two arms on one build under two
policies are two run configs.

Rules, and the measurement behind each:

- **Precedence per label.** Source keys are `kuzmin2018`, `kuzmin2020`, `costanzo2016@30`,
  `costanzo2016@26` and `converted_zero`. Costanzo splits by temperature because its two screens
  are separate experiments; Kuzmin does not, because it screened at one. Each screen's stored
  interaction is reproduced by its own fitness and by little else, so the precedence is what
  decides whose standard a model learns.
- **The converted 0 yields to any measurement.** SGD essentiality and a SynthLethDB lethal pair
  both arrive as fitness 0. Averaging one into a measured mean is hazard H1, which pulled 907
  singles of the 025 build down by a median of 0.16.
- **Same-source replicates combine by inverse variance** by default, which cannot be less certain
  than the better single measurement. The 025 build used a root-mean-square of the standard
  deviations, which is not that.
- **Stouffer over the source p-values**, with the score's sign restoring the direction a two-sided
  p does not carry. Two screens that disagree about the sign of an interaction combine toward a
  larger p, which is the honest reading.
- **`source_convention`.** The released Kuzmin trigenic scores set the query single-mutant fitness
  to 1, and that reproduces the published value on 99.98 percent of 2018 rows and 99.56 percent of
  2020 rows. The supplementary methods instead weight the two control terms by the measured query
  singles, and Kuzmin 2018 releases those on 99.2 percent of its control rows. The two are
  different numbers by construction, so a run names which it takes rather than averaging them.
- **`prefer_strain_matched_double`.** A triple's identity wants the fitness of the double-mutant
  query strain its own screen used, not the pair's digenic array screen. On Kuzmin 2018 those two
  agree at r 0.777 with a median absolute difference of 0.045, more than half the 0.08 calling
  threshold.

The join key is the `tm` token, not the whole strain identifier, because the two years write it
differently: 2018 puts the full pair on both query perturbations (`YKL010C+YMR067C_tm2424`), 2020
puts it on one (`YDR003W_tm1501`) and leaves the other bare, and the double-mutant query strain's
own fitness record carries the full pair form. Measured on 5,000 triples of the 029 build, the
token resolves in all 5,000.

Until the Kuzmin query-strain fitness records reach the served graph, that rule falls through to
the pair's digenic screen, which is the 0.52 the 029 closure recompute reaches on the Kuzmin 2018
screen. That is expected rather than a defect: the records exist in the loaders only as of
4c4a4f950 and are not in any build yet.
