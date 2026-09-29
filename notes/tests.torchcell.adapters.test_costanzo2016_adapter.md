---
id: y17x6vaf531w8l70e1l0qae
title: Test_costanzo2016_adapter
desc: ''
updated: 1705536044369
created: 1705534722997
---
## test_no_duplicate_warnings

The idea behind this test is to provide some quality assurance on the datasets. By restricting certain types of warnings I think we are more likely to face less issues, and we will have less to debug because we will know that if there are  duplicates in the database it is due to double entry by two separate datasets. It is always easier to relax constraints later.

## 2026.09.28 - Exact graph emission for the Costanzo confs, with real chunking (Phase 9)

3 tests over the Costanzo 2016 confs through the shared harness ([[tests.torchcell.adapters.test_kuzmin2018_adapter]]): `dmf_costanzo2016` and `dmi_costanzo2016` differ from the Kuzmin confs only by `memory_reduction_factor: 0.5` on the publication node and the nine chunked edges, and with two records `int(2 * 0.5) = 1` makes those methods run as two one-record chunks, so the factor path is exercised rather than read back; a missing conf is refused before wandb starts; `main` builds the DMI 5e5 subset and writes everything (timestamped directory matched by pattern). Coverage 21% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
