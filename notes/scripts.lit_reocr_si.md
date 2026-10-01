---
id: swlmwrzij3negismq9y7sm5
title: Lit_reocr_si
desc: ''
updated: 1790876234808
created: 1790876234808
---

## 2026.10.01 - Re-OCR SI PDFs into per-PDF figure directories (issue #579)

Repairs a mirror key whose SI figures were lost to the shared `si/images/` of the old runner. Order, so nothing is retired before the OCR has succeeded: OCR every `si/si*.pdf` of every `--key` in natural order (`paper.pdf` is not re-OCR'd); refuse with `UnresolvedFigureError` if any SI markdown still references a missing figure; retire the flat pre-#579 `si/images/<file>` figures through `scripts/deprecate.sh` under the per-key name `<key>__si-images-flat-pre579` (two keys deprecated in the same second no longer collide on the basename `images`); force-backfill each manifest from Zotero (read only). Loads the repo `.env` by path, so `DATA_ROOT` (mirror and MinerU model cache) and the Zotero credentials are set wherever it runs. Launch line (one GPU, slurm) is in the module docstring. Tested by [[tests.scripts.test_lit_reocr_si]].

## 2026.10.01 - Delta review: phases, Zotero first, safe retirement

Phases now run over every key before the next starts: Zotero index (a key absent from it raises `KeyNotInZoteroError` before any OCR), OCR, reference check, retirement, a second reference check (a markdown that pointed at a flat figure dangles after retirement and refuses by name; the operator moves the named file back from the logged graveyard entry and investigates), backfill (anything but `enriched` raises `NotEnrichedError`). `deprecate.sh` now gets the full environment, so its refusal of a graveyard inside `DATA_ROOT` applies; on any `deprecate.sh` failure the staged figures are moved back and `RetireError` carries the exit code and stderr. A staging directory left by a killed retirement is retired on the next run. Six mutants (backfill before retire, per-key phases, check only the first key, retire only the first key, `main` ignoring `--graveyard`, `DATA_ROOT` stripped) each fail the tests.
