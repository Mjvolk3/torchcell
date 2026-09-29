---
id: 7v8guay3sva14znszfbdbgi
title: Viz
desc: ''
updated: 1790652070759
created: 1790652070759
---

## 2026.09.28 - boxplot tick labels for matplotlib 3.11

`colony_shape_by_volume_panels` called `boxplot(labels=...)`, deprecated in matplotlib 3.9 and removed in 3.11; CI resolves the requirements floor to 3.11 and the call raised `TypeError` there while the local 3.10.7 only warned. The call now uses `tick_labels=`, available since 3.9, and the floor in `env/requirements.txt` moved from 3.7.2 to 3.9. Found by [[tests.torchcell.sga.test_viz]] in Phase 9 of [[plan.test-suite-buildout.2026.09.25]]; the one source change of that phase.
