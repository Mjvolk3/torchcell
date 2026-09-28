---
id: 7lwmang8rkjvkgknxy4yj7u
title: Test_smith2006_synthetic
desc: ''
updated: 1790562905536
created: 1790562905536
---

## 2026.09.27 - The Smith 2006 loader built end to end

Seven tests. Nothing installed can write a BIFF `.xls` (no xlwt; xlrd 2 only reads), so the one-line loader `_read_table` is stubbed and everything after it runs for real, download and deposit included. Loader coverage 95%. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
