---
id: gyv9n7zh5vr7z3ktaf476m3
title: Test_run_mineru
desc: ''
updated: 1790872727746
created: 1790872727746
---

## 2026.10.01 - Per-PDF figures with MinerU faked (issue #579)

Fake `mineru` modules in `sys.modules`; the fake `do_parse` writes MinerU's nested tree from a scripted figure list per stem. Asserted with exact file lists and exact markdown and content-list strings: two SI PDFs sharing `si/` both keep their figures under `images/si1/` and `images/si2/`; a re-run of si1 replaces only its own directory (si2's and the flat legacy `images/legacy.jpg` are byte-identical), and a run with no figures removes only si1's directory; `paper.pdf` keeps the flat `images/` and unchanged references; a reference to an unwritten figure exits 5 with the exact stderr line and writes nothing; the stdout facts `MINERU_VERSION=2.7.6` and `MINERU_DPI=200` / `350`, and the DPI the page loader saw.
