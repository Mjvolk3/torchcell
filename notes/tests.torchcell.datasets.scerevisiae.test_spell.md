---
id: 9y6faxgb48reu7hcn3ooiyo
title: Test_spell
desc: ''
updated: 1791321230698
created: 1791321230698
---

## 2026.10.06 - Phase 22 behavioral tests for the SPELL loader

Test file: `tests/torchcell/datasets/scerevisiae/test_spell.py` (122 tests), target [[torchcell.datasets.scerevisiae.spell]]. Hermetic: hand-written PCL files and zip archives under `tmp_path`; no network, no `DATA_ROOT`, no real SPELL archive.

### Fixtures

- `_PCL`: three genes (YAL001C TFC3, YAL002W VPS8, YAL003W EFB1) by three conditions (`WT_30C_10min`, `WT_37C_10min`, `WT_42C_10min`); EWEIGHT row `1, (empty), 2`, so `eweights = [1.0, 1.0, 2.0]`; one empty expression cell (YAL002W, `WT_37C_10min`) that reads as NaN. GWEIGHT parses as int64.
- `_PCL_B`: YAL001C only, conditions `c1 = 2.0`, `c2` empty.
- Condition strings for the eight parsers come verbatim from `experiments/015-spell/results/spell_knockout_conditions.csv` (2,804 distinct `condition_name` values) wherever a real example of the branch exists; the rest use the pattern written in the source comment and carry `synthetic` in the test id.

### Expected values

- Extraction confidence is `min(0.9, 0.1 + k / 14 * 0.8)` rounded to 3: k=0 -> 0.1, 1 -> 0.157, 2 -> 0.214, 3 -> 0.271, 6 -> 0.443, 14 -> 0.9.
- The six-row export (two studies) has hand-derived rows; mean confidence 1.113 / 6 = 0.1855, printed `0.185` because the float is 0.18549999.
- Global histogram: 8 non-NaN values, sum 2.1, median (0.25 + 0.45) / 2 = 0.35; stats text checked against `statistics.fmean`, `statistics.median`, `statistics.pstdev`.
- Plot tests record `Axes.hist` (exact arrays and kwargs) and read titles, texts and `get_suptitle()`; `plt.show` and `plt.savefig` are stubbed.

### Findings (pinned as the code behaves)

- spell.py:48: the EWEIGHT row is `.strip()`ped before `split`, so a trailing empty weight is dropped and `eweights` is shorter than `conditions`.
- spell.py:46-51: row 2 is taken as EWEIGHT without checking its label; a PCL without that row loses its first gene, whose values become the weights.
- spell.py:55-58: conditions are `header[3:]` by position; a GID-first PCL reports `GWEIGHT` as a condition.
- spell.py:424: the `N C` temperature pattern runs case-insensitively, so ORF suffixes (`ykl020c deletion` -> 20.0) and the strain `S288C` (-> 288.0) read as Celsius; 278 of 2,804 real labels get a temperature, 272 through an ORF suffix. The real form `HHO1_delta_37_degrees` gets None because `\s*deg` does not accept `_`.
- spell.py:494: the concentration unit alternation includes `M` under `re.IGNORECASE`, so `20min` is concentration 20.0 with unit `m` (138 real labels), which also sets `needs_manual_review`.
- spell.py:566: the pH pattern has no word boundary, so gph1, rph1, pph3, pph21, pph22 read as pH 1, 1, 3, 21, 22 (14 real labels); the second, parenthesized pH pattern is unreachable because the first matches every string it matches.
- spell.py:622, 624: bare `g1` / `g2` keywords tag glg1, hog1, mig1, glg2 as cell-cycle phases (102 real labels).
- spell.py:658: the trailing-number replicate pattern makes bare gene names (`erg6`) replicate 6 of type `unknown`.
- spell.py:384-397: time units are tried in the order min, hr, sec, not by position, so `2 hr 30 min` is 30.0; a bare `h` (`t=1h`) is not parsed although the module note lists `4h`.
- spell.py:718, 734, 761: category keywords are bare substrings (`hs` makes hsp12 heat_shock, `hypo` makes hypoxic osmotic_stress, `dna` makes a genomic-DNA reference note dna_damage).
- spell.py:843: the comment says 5 fields give 0.5; the formula gives 0.386.
- spell.py:25: module `DATA_ROOT` is a hard-coded `~/Documents/projects/torchcell`, and the `DATA_ROOT` environment variable is never read; the default export path is `<that>/data/sgd/spell/spell_conditions_metadata_enhanced.csv`, not the docstring's `DATA_ROOT/spell_conditions_metadata.csv`.
- spell.py:222: `studies_to_load` is a substring test on the full zip path, so a token in the root directory name selects every archive.
- spell.py:226, 229: `max_studies=0` is falsy and means unlimited.
- spell.py:342: the per-gene plot title counts every loaded study, not the studies that measured that gene.
- spell.py:1043: `check_condition_metadata_quality({})` divides by zero.

### Left uncovered

`main` (spell.py:1082-1169, out of scope for the phase) and the `plt.show()` branch of `plot_global_expression_distribution` (spell.py:192). Coverage of `spell.py` from this file: 90% (455 statements, 54 missed).

## 2026.10.06 - Audit 1 revisions

- `test_extract_nutrient_info` now pins first-entry priority in all three dicts: "ammonium to proline shift" is ammonium and "C-lim to N-lim shift" is carbon_limited, although each string ends on the later entry.
- `test_extract_replicate_info` adds "bio rep 2" -> (2, "biological"), so the `bio rep` branch of the type check is exercised.
- `test_second_ph_pattern_is_subsumed` reads both pH patterns from `extract_physical_params.__code__.co_consts` instead of copying them. Correction: pattern 1 always matches first, but not always on the same token; "gph1 (pH 7)" returns 1.0 (from gph1), not 7.0. The row is also pinned in `test_extract_physical_params` as a Finding.
- The missing-YORF `KeyError` match is anchored: `str(KeyError)` is the quoted message `"None of ['YORF'] are in the columns"`.
- The `studies_to_load` test also compares the loaded Beta frame, not only the keys.
- Correction to the replicate Finding (spell.py:658): 440 of the 2,804 distinct real labels take their replicate number from trailing digits alone (no `#N`, no `rep N`); 135 of those are bare gene names.

### Reach of the findings

The parser and category findings (temperature, concentration unit, pH, cell-cycle phase, replicate number, time units, category substrings) reach real outputs: `experiments/015-spell/scripts/run_phase1_spell_analysis.py` calls `export_condition_metadata` to write `spell_conditions_metadata_enhanced.csv`, and `experiments/015-spell/scripts/spell_coverage_analysis.py` reads it into `spell_coverage_report.md` (temperature ranges, concentration by unit, pH ranges, cell-cycle fields, `primary_category` counts, `needs_manual_review`). Those outputs are cited by [[experiments.015-spell.scripts.run_phase1_spell_analysis]], [[experiments.015-spell.scripts.spell_analysis]] and [[experiments.015-spell.scripts.spell_coverage_analysis]]. The remaining findings (hard-coded `DATA_ROOT`, the substring study filter, `max_studies=0`, the per-gene plot title, the empty-input division by zero) are latent: no recorded output depends on them.
