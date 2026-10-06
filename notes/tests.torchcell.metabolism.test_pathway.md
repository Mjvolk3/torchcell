---
id: j67vwnw5f6mgtfitugk14x4
title: Test_pathway
desc: ''
updated: 1791270265576
created: 1791270265576
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): every refusal of `apply_pathway` and `_derive_formula_and_charge` with its exact message (metabolite clash, no producing reaction, a product coefficient of 2.0, a negative element count `{'H': -1, 'N': -1}`), `int()` truncation on a fractional coefficient (0.7 a_c derives C2H4O, then the balance check refuses), a gene rule plus evidence annotation edited in place, and `product_ids` ignoring demands and intermediates.
