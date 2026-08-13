# Run 4 -- single + double KO panel, 3 re-randomized plates (2026-08-06)

First round of the assay carrying **double** knockouts, so this is the first data
from which a **digenic interaction** (epsilon) can be computed rather than only
single-mutant fitness.

## Provenance

| field | value |
|---|---|
| Constructed + plated by | Aurosish Sharma |
| Echo run date | 2026-08-06, 16:28:02 / 16:33:07 / 16:37:29 (P1 / P2 / P3) |
| Echo Run IDs | 1786055592 (P1), 1786055898 (P2), 1786056160 (P3) |
| Instrument software | Echo Cherry Pick 1.8.3, `Protocol 1.ecp` |
| Source pick lists (instrument) | `E:\TorchCell_ECHO\20260806_sKO_dKO_Plating\TC_sKO-dKO_OD1_5nL_P{1,2,3}.csv` |
| Handed over | 2026-08-11, to `/tmp/screenshots/double-kos` on GilaHyper |
| Retrieval method | `manual_handoff` -- files copied from the collaborator's drop, not downloaded |
| Integrity | `SHA256SUMS.txt`, all 11 files verified on copy into the repo |
| Transfer volume / density | 5 nL at OD 1.0 |
| Incubation at imaging | **48 h 12 min** (officially recorded; not the nominal "48 h") |
| Images | 4712 x 3534 JPEG, same backlit rig as run 3 (EXIF portrait; the plate crop in `preprocess_fullres` finds the bright plate regardless of camera orientation) |

`Single-and-Double-KO-Strains-List-Order.numbers` is the collaborator's original
Apple Numbers file, retained verbatim; `...-List-Order.csv` is the export the
loaders read. Both are kept so the export can be re-derived and checked.

## Design

**26 strains x 3 independently randomized plates.** Per plate: 378 transfers into
384 wells (6 blanks), as

- WT (BY4741) -- **28** replicate wells
- 12 single KOs (`s1`..`s12`) -- **14** wells each
- 13 double KOs (`d1`..`d13`) -- **14** wells each

so 42 colonies per mutant strain and 84 WT wells across the three plates. Each
plate has its own randomized layout (`TC_sKO-dKO_OD1_5nL_P{1,2,3}.csv`), which is
what lets a bootstrap across plates average out position/plate batch bias -- the
same design principle as run 3.

**Every double has both of its constituent singles on the same plate.** That makes
`epsilon = f_ab - f_a * f_b` computable entirely within a plate, with no
cross-plate normalization in the interaction term.

## Two facts that will bite an analysis if missed

1. **`YKL033W-A x YJR060W` (CBF1) failed to construct.** Reported by Aurosish on
   handover. It is absent from the strain list -- there are 13 doubles, not the 14
   the round was designed around, and the missing pair is not a dropout to be
   imputed. See `[[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]]`
   for the selection that produced the intended 14.

2. **`s7` is `YLR104W` (LCL2), NOT `YPL056C` (LCL1).** Run 3's panel carried
   LCL1/YPL056C; this panel carries LCL2/YLR104W. The single-mutant panels
   therefore differ by one strain, so any run-3-vs-run-4 single-mutant comparison
   must reconcile that rather than joining on position in the list.

## Files

```
P{1,2,3}_sKO-dKO_OD1-5nL.JPG          plate images, 48 h 12 min
P{1,2,3}_Transfer_Report.csv          Echo per-transfer report (actual volumes, status)
TC_sKO-dKO_OD1_5nL_P{1,2,3}.csv       Echo cherry-pick lists (well -> strain)
Single-and-Double-KO-Strains-List-Order.csv       strain id -> KO1/KO2
Single-and-Double-KO-Strains-List-Order.numbers   collaborator original
SHA256SUMS.txt                        integrity manifest
```
