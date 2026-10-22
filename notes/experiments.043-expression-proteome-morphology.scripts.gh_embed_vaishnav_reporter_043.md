---
id: jthph871lz3k7luxksdg060
title: Gh_embed_vaishnav_reporter_043
desc: ''
updated: 1792637971590
created: 1792637971590
---

## 2026.10.22 - Launcher

GPU stage of [[experiments.043-expression-proteome-morphology.scripts.embed_vaishnav_reporter_043]]: `--stage embed` then `--stage verify`, one card, 32 GB, 1.5 h. Submitted as job 6115 with `--dependency=afterany:6106` (the isolate four-encoder pass). The 1.5 h rests on an assumed FUDT upstream rate of 150 to 300 windows per second (hypothesis; 6106 measures it first).

Update, same day: job 6106 measured FUDT upstream at 73.9 windows per second (76,800 windows in 1,040 s, `043-iso-embed_6106.out`), half the low end of the assumed rate. 71,158 windows then take about 16 min, inside the 1.5 h limit.
