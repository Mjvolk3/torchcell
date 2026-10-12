---
id: yicgi6ia8cgt592r4ydvzvv
title: _kuzmin2018_s4
desc: ''
updated: 1791815297354
created: 1791815297354
---

## 2026.10.12 - A synthetic Additional Data S4 for the Kuzmin 2018 builds

`install(monkeypatch, tmp_path, rows)` writes an S4 workbook with the released columns into a throwaway raw mirror, points `kuzmin2018.raw_mirror_dir` at it and pins `kuzmin2018.S4_SHA256` to its bytes, so `download()` links it as it links the real file. Used by `test_kuzmin2018.py` and `test_kuzmin2018_synthetic.py` (an autouse fixture each) since #1054 made Smf and Dmf read S4.
