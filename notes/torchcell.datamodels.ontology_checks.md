---
id: c4ancndhl83xcaaa6rn3kf0
title: Ontology_checks
desc: ''
updated: 1790872937989
created: 1790872937989
---

## 2026.10.01 - Unreadable dict keys; no-compound records

- `_dict_keys` returns None (unreadable, the outcome `AdapterNodeSite.property_keys` already documents) for a dict literal with a non-literal key, instead of `[]`, which read as "emits no properties" and made `adapter_property_mismatches` report every declared property as never emitted. On the real `cell_adapter.py` the site list and the mismatch list are byte-identical before and after (no real site has a computed key).
- `JoinKeyCensus` gains `n_with_no_compounds`; a record naming no compound is counted there, not in `n_with_every_compound_identified`. `join_key_audit` has no production caller (tests and the API docs page only), so no committed report reads differently.
- Issue #536; tests `test_a_computed_key_reads_as_unreadable_and_is_skipped`, `test_join_key_audit_counts_each_partial_record_exactly`.
