---
id: gyv9n7zh5vr7z3ktaf476m3
title: Test_run_mineru
desc: ''
updated: 1790872727746
created: 1790872727746
---

## 2026.10.01 - Per-PDF figures with MinerU faked (issue #579)

Fake `mineru` modules in `sys.modules`; the fake `do_parse` writes MinerU's nested tree from a scripted figure list per stem. Asserted with exact file lists and exact markdown and content-list strings: two SI PDFs sharing `si/` both keep their figures under `images/si1/` and `images/si2/`; a re-run of si1 replaces only its own directory (si2's and the flat legacy `images/legacy.jpg` are byte-identical), and a run with no figures removes only si1's directory; `paper.pdf` keeps the flat `images/` and unchanged references; a reference to an unwritten figure exits 5 with the exact stderr line and writes nothing; the stdout facts `MINERU_VERSION=2.7.6` and `MINERU_DPI=200` / `350`, and the DPI the page loader saw.

## 2026.10.01 - Review fixes

Added the crash re-run tests (failed copy; kill between the moves), the `SOM.pdf` then `paper.pdf` re-run test, anchored reference forms (markdown, HTML, content list, uppercase and spaced names, prose URL untouched), and the exit-5 test now asserts the PDF's directory holds only the PDF (no scratch). Each failed on the previous runner.

## 2026.10.01 - Kill mid-swap then a failing re-run

Parametrized over the recorded phase (`retiring`, `installing`, `swapped`, none): a kill parks `old.jpg` in `.images.old`, the next run exits 5, and the old markdown is unchanged with every figure it references back in `images/si1` (arrivals of an `installing` kill removed, both sets after `swapped`), and no scratch remains. The header no longer claims a failed swap always leaves the previous markdown with its figures.

## 2026.10.01 - Real kills

Replaced the hand-built post-kill states with `_Killer`, which kills the k-th `Path.rename`, `Path.write_text` (after writing half its text), `os.replace`, `shutil.copytree` or `shutil.rmtree` of the real runner. `test_any_single_kill_...` and `test_any_double_kill_...` loop over every kill point (and every pair) followed by an exit-5 run and assert every reference in the markdown and content list resolves; `test_a_kill_during_install_then_during_recovery_keeps_the_old_figures` pins the reviewer's double-kill sequence with exact file lists.

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `_ensure_hf_home` on all three branches (kept, derived from `DATA_ROOT` as `models/mineru/hf_cache`, exit 4 with the exact stderr line), `_find_first` returning None, and the early exits of `main`: 2 for a missing PDF (nothing created), 4 before MinerU runs, 3 when MinerU writes no markdown (the scratch directory stays, unlike exit 5), and a markdown-only MinerU tree that writes just the markdown.
