---
id: ik4sk36vnjaotlbt2hw5azs
title: Test_betaxanthin
desc: ''
updated: 1791270257847
created: 1791270257847
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the betaxanthin cassette against a nine-species stand-in for yeast-GEM 9.0.2 (charged formulas: NADPH -4, NADP+ -3, glutamate -1, lysine +1). Pinned: the partner filter and its warning (16 of 20 absent), the exact reaction and metabolite order, stoichiometries, gene rules, spontaneity, the evidence census (convention 3, sourced 6, derived 5; 2, 4, 4 without the oxidase branch), the demand ids, and the derived formulas after `apply_pathway`: alanine C12H14N2O6 (0), glutamate C14H15N2O8 (-1), lysine C15H22N3O6 (+1), tyrosine C18H18N2O7 (0), betanidin C18H16N2O8 (0), each betalamic acid C9H9NO5 + partner - H2O.

Finding: `HeterologousPathway.product_ids` says it returns one species per condensation partner for betaxanthin, but on the cassette it also returns water `s_0803`, NADP+ `s_1207` and betanidin (pathway.py:137-148). No caller reads it yet.
