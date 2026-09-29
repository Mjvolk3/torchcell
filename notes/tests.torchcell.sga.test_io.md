---
id: glpv12v031aady6wpxp74h8
title: Test_io
desc: ''
updated: 1790649934361
created: 1790649934361
---

## 2026.09.28 - Gitter tables, Echo picklists and layout merges (Phase 9)

7 tests (12 cases): well labels decode as bijective base 26 (Z = 26, AA = 27) with the digits as the column, accepting lowercase and blanks and refusing digit-first or empty labels; `read_gitter_dat` skips comment lines, mixes tab and space separators and flags a four-field row; `read_echo_picklist` decodes destination wells to (row, col), keeps the sample name as a string and coerces transfer volumes, naming the absent required columns sorted; `merge_layout` left-joins on row and column so every colony survives, a colony without a layout entry carrying NaN strain, volume and well, and without a layout it adds null columns without mutating its input. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
