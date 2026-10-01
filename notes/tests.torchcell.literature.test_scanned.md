---
id: evrui53ikuxfvhhl82ms2gb
title: Test_scanned
desc: ''
updated: 1790876260040
created: 1790876260040
---

## 2026.10.01 - Pass records

The fake `ocr_pdf` now writes what the real one writes (`<stem>.md` and `<stem>_ocr_provenance.json`, overwritten per pass). Added: the final record is the last pass's record with `params["passes"]` holding both full records (whole JSON compared), and an empty DPI sweep refuses with the exact message.
