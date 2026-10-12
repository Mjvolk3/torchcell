---
id: eix6goezjdfbef8vwoyec6p
title: Kinetic_measured_tier_audit
desc: ''
updated: 1789100296962
created: 1789100296962
---

## 2026.09.10 - Is the measured tier organism-pure, on the right substrate, and as big as it could be

Four questions about the 148 catalytic units the Open Enzyme Database prices: the organism of
every row a value came from, whether the row was assayed on a substrate the reaction
consumes, the reach of four alternative match keys, and the measured-versus-predicted
disagreement in decades.

Measured 2026.09.10: the tier is 100 percent *S. cerevisiae* (the full 35,753-row database
holds exactly the mirror's 1,126 rows for the organism); 74 of 148 units took their value
from a row assayed on a substrate the reaction does not consume; an EC-plus-substrate key
across organisms would reach 866 units against the accession key's 148; the largest
measured-versus-predicted gap is 1.94 decades and none exceeds two.

Writes `results/kinetic_measured_tier_audit.json`. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - Rebuilt on the shipped resolver, plus the homolog contest

The audit no longer reimplements the resolution cascade: it calls `_database_tiers` three
times, once with the substrate check disabled to reproduce the old accession-only key, once
with it on, and once with the homolog mirror alone. That third reading is what makes the new
result possible: on the 87 units with a substrate-matched yeast measurement plus both a
homolog value and a prediction, the homolog is closer on 55, median error 0.094 decades
against the prediction's 0.300.

Measured 2026.09.11: 148 units under the accession key, 96 substrate-matched; of the 52
removed, 33 go to the homolog tier, 18 to predicted, 1 to default.
