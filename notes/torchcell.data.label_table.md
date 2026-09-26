---
id: len2q6isd3e846rs2eg9oi7
title: Label_table
desc: ''
updated: 1789963201367
created: 1789963201367
---

## 2026.09.21 - Applying a policy to a build, and recovering the roles a record does not label

`build_label_table` is one pass over a no-merge build's processed LMDB: one row per record naming
the chosen value, its source, its combined uncertainty and p-value, how many entries were available
and how many were combined. It caches to `<build_root>/label_tables/<policy_id>.parquet`, beside
the split caches, so a rule change is a small new file.

`triple_roles` is the part that was not obvious. The trigenic identity is asymmetric: one gene came
from the array and the other two were crossed in as a double-mutant query strain, and the score
subtracts the query double's fitness times the array single. No record labels those roles. The
strain identifiers give them away: exactly one perturbation carries an array strain (`_dma` or
`_tsa`), and the other two carry the query strain's `tm` token. Measured on 5,000 triples of the
029 build, both resolve in all 5,000, across both Kuzmin years, whose loaders write the identifier
differently. Without this the identity can only be evaluated in its symmetric form, which is what
the 029 closure recompute was limited to.

The function returns None when the roles are ambiguous rather than guessing, which is the honest
answer for a record whose source never named its strains.
