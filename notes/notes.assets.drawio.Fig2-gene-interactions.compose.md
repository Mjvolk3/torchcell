---
id: mureg3ybd1eogsnkt4q2idn
title: compose
desc: ''
updated: 1791910907295
created: 1791910907295
---

## 2026.10.13 - Composing Figure 2 headless

`Fig2-gene-interactions.compose.py` places the three generated panels (d additive null, e graphs help, f training pool) into the row-2 slots of `Fig2-gene-interactions.drawio.svg` at true size through `drawio_true_size_import.image_cell`, clears the sketch placeholders, drops the row-3 placeholders i and j, and relabels the letters so the figure reads a to g (g is the nested cell). Idempotent: rerunning refreshes the three payloads. Export is the draw.io desktop .deb under xvfb (memory `drawio-headless-export-gilahyper`): `drawio --no-sandbox --disable-gpu --disable-dev-shm-usage --export --format pdf --crop --output <pdf> <plain .drawio>`; the AppImage build refuses the input with "input file/directory not found", and the exporter wants the plain mxfile XML, which `load_drawio` extracts from the `.drawio.svg`. Panels b and c are the hand-placed images from before and were not touched; updating them to the 025 S3 model waits on a test evaluation of its checkpoints.
