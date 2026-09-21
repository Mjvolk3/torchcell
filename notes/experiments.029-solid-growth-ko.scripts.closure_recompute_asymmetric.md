---
id: mc97tz4zoe8jh0z4ttumetj
title: Closure_recompute_asymmetric
desc: ''
updated: 1789964180475
created: 1789964180475
---

## 2026.09.21 - The published identity, evaluable on 029 after all

`closure_recompute.py` evaluates the SYMMETRIC trigenic identity,

    tau = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k

because a record does not say which of its three genes came from the array, and without
that the asymmetric form cannot be written down. It reaches r = 0.517 on the Kuzmin 2018
screen under a Kuzmin-first policy.

The roles turn out to be recoverable. Exactly one perturbation of a triple carries an
array strain (`_dma` or `_tsa`) and the other two carry the query strain's `tm` token, so
`torchcell.data.label_table.triple_roles` returns the array gene, the query pair and the
query strain id. Measured here on the whole build, roles resolve for **299,146 of 299,146
triples**. That makes the published form evaluable:

    tau = f_ijk - f_ij f_k - eps_ik - eps_jk

the triple's own fitness, minus the double-mutant query strain's fitness times the array
single, minus the two single-mutant control queries' adjusted scores against that same
array gene, with the query singles entering as 1, which is what the released scores do.

**What this separates.** The symmetric form differs from the published one in shape AND
in which measurements it draws on: it uses all three doubles and all three singles, where
the published form uses one double, one single and two control epsilons. Running both on
the same records under the same policy tells us which of the two is costing us. If they
score about the same, the remaining gap is the one term the build cannot supply, the
double-mutant query strain's own fitness, which falls through to the pair's digenic array
screen because those records enter the loaders only as of 4c4a4f950 and are in no build
yet. If the asymmetric form scores clearly better, the shape was costing us too.

Values are chosen by `LabelPolicy` under a Kuzmin-first precedence with the triple's own
Kuzmin year promoted, which is the same rule the symmetric recompute used, so the two
numbers are like for like.

### Result

| screen | form | n | r | rho | slope | rmse |
|---|---|---|---|---|---|---|
| Kuzmin 2018 | asymmetric, published | 57,451 | 0.511 | 0.424 | 0.92 | 0.079 |
| Kuzmin 2018 | symmetric | 57,451 | 0.504 | 0.428 | 0.91 | 0.079 |
| Kuzmin 2018 | asymmetric, array single forced to 1 | 57,451 | 0.427 | 0.346 | 1.23 | 0.146 |
| Kuzmin 2020 | asymmetric, published | 236,225 | 0.325 | 0.384 | 0.93 | 0.165 |
| Kuzmin 2020 | symmetric | 231,611 | 0.278 | 0.337 | 0.90 | 0.212 |
| Kuzmin 2020 | asymmetric, array single forced to 1 | 236,442 | 0.273 | 0.287 | 0.89 | 0.212 |

**On Kuzmin 2018 the shape of the equation is not what costs us.** The published form
scores 0.511 against the symmetric form's 0.504 on the very same 57,451 records under the
same policy, a difference of 0.007. I predicted about 0.538, the within-screen figure with
only the query double fitness degraded, and measured 0.511; the two substitutes for that
term differ, so the bands agree without the numbers matching. The reading stands: on this
screen the entire remaining gap is the one term the build cannot supply, the
double-mutant query strain's own fitness, which falls through to the pair's digenic array
screen. That is a cleaner attribution for the thesis chapter than the symmetric number
alone supports, because it rules out the formula shape as an explanation.

**On Kuzmin 2020 the shape does matter, and the reason is the convention.** The published
form scores 0.325 against 0.278, and its error drops from 0.212 to 0.165. Kuzmin 2020
releases a query fitness on only 0.4 percent of its control rows and its methods say every
missing fitness was assigned 1.0 during scoring. The published form needs no query singles
at all, since they enter as 1, so it matches what the source did; the symmetric form needs
all three singles and is forced to reconstruct values the source never used. The screen
where the source substituted 1.0 is exactly the screen where using the source's own form
pays.

**The array single is a real term.** Forcing it to 1 as a control costs 0.084 on 2018 and
0.052 on 2020 and wrecks the slope on 2018 (1.23), so the identity is not insensitive to
its terms; the query double is simply the one that is missing.

The asymmetric form is also computable on more records, 236,225 against 231,611 on 2020,
because it needs one double and one single where the symmetric form needs three of each.

**Consequence.** `source_convention="published"` is the right default for `LabelPolicy`,
which is how it ships. The open test is the 030 build: with the double-mutant query strain
fitness present as a record, the strain-matched rule stops falling through, and the
question is whether 0.511 moves toward the 0.99 the within-screen reconstruction reaches.
