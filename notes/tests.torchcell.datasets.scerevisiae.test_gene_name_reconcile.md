---
id: yvhob0wc5sfrmmxijl2emy5
title: Test_gene_name_reconcile
desc: ''
updated: 1790882021840
created: 1790882021840
---

## 2026.10.01 - Real genome only when its database is trusted

The data-gated genome construction now first calls `require_trusted_genome_database` (see [[tests.torchcell.conftest]]): when the real `data.db` would be built or migrated, the test fails by name instead of migrating the shared root.
