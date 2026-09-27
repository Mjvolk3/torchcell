---
id: rkveiu4bdgv0k1xtljnmjdi
title: Test_yeastphenome_synthetic
desc: ''
updated: 1790482277555
created: 1790482277555
---

## 2026.09.26 - Hermetic synthetic-screen tests for the YeastPhenome loader

Test file: `tests/torchcell/datasets/scerevisiae/test_yeastphenome_synthetic.py` (13 tests, 16
cases with the parametrized readout parser), covering
[[torchcell.datasets.scerevisiae.yeastphenome]]. Complements the data-gated mirror test
[[tests.torchcell.datasets.scerevisiae.test_yeastphenome]], which is unchanged. Part of the
test-coverage campaign ([[plan.test-suite-buildout.2026.09.25]]); the module sat at 26.6% line
coverage before this file.

### Approach

`yeastphenome.SCREENS` (44 pinned real screens) is monkeypatched to two synthetic entries and
each screen's `<pmid>_<stem>_valuez.txt` is hand-written under `<root>/raw/`, so PyG skips
`download()`. `raw_file_names` reads the module global at call time, which is what makes the
patch take effect (pinned by its own test). Compound identity comes from the in-repo table
through `resolved_compound`, so no network is touched.

### Fixture

Screen A (PMID 99999901, stem `synthetic_a`), one header cell per grammar branch:

| col | header | fate |
|---|---|---|
| 1 | `hom \| growth (colony size) \| furfural [10 mM] \| YPD \| synthetic` | kept, BY4743 diploid, YPD solid |
| 2 | `hap a \| growth (culture turbidity) \| quinine [2 mM] \| SD \| synthetic` | kept, BY4741 haploid, SD liquid |
| 3 | `het \| growth (colony size) \| furfural [10 mM] \| YPD \| synthetic` | het_or_ambiguous_zygosity |
| 4 | `hom \| growth (colony size) \| heat [37 C] \| YPD \| synthetic` | condition_unparseable (unit C) |
| 5 | `hom \| growth (colony size) \| furfural [10 mM] \| synthetic` | media_absent_or_unknown (4 fields) |
| 6 | `hom \| expression (microarray) \| furfural [10 mM] \| YPD \| synthetic` | not_growth |
| 7 | `hom \| growth` | malformed_header |
| 8 | `hom \| growth (colony size) \| furfural [10 mM] \| LB \| synthetic` | media_absent_or_unknown (LB) |
| 9 | empty cell | skipped silently |

Rows: `YAL001C 1.5 / -2.25`, `YBR002C 0.5 / (blank)`, `not_an_orf 3.0 / 3.0`, `Q0250 -0.75 /
0.125`. Screen B (PMID 99999902, stem `synthetic_b`): one column `hap alpha | growth (pooled
barseq) | furfural [10 mM] | SC + EtOH | synthetic`, one row `YAL001C 2.0`.

### Expected values asserted

- Six records in write order: A col 1 -> idx 0-2 (YAL001C, YBR002C, Q0250), A col 2 -> idx 3-4
  (YAL001C, Q0250; the blank YBR002C cell yields no record), B col 1 -> idx 5.
- Each record compared by whole `model_dump()` equality against a hand-built
  `EnvironmentResponseExperiment`: `KanMxDeletionPerturbation(orf, orf)`; `Environment(media,
  temperature=None, perturbations=[SmallMoleculePerturbation(compound, Concentration(value,
  millimolar))], provenance_gaps=[temperature gap anchored to CURATION])`;
  `EnvironmentResponsePhenotype(z_score, npv, units=NPV_UNITS + " [readout: <method>]",
  gaps n_samples / environment_response_uncertainty / sample_unit)`. References carry NPV 0.0,
  `ReferenceGenome(strain, ploidy)` from `_ZYGOSITY`, and the same environment. Publications
  are `Publication(pubmed_id=<pmid>, pubmed_url=...)` per screen.
- Furfural pins to `Compound(name furfural, inchikey HYBBIBNJHNGZAN-UHFFFAOYSA-N, smiles
  C1=COC(=C1)C=O, pubchem_cid 7362, chebi_id CHEBI:30976)`.
- `gene_set.json == ["Q0250", "YAL001C", "YBR002C"]`; no `data.csv` (`ds.df is None`);
  reference index `[[0, 1, 2], [3, 4], [5]]` with strains BY4743 / BY4741 / BY4742.
- Final log line for screen A alone: `Wrote 5 YeastPhenome records over 2 environments (1
  screens); dropped columns/rows: {'non_orf_row': 2, 'het_or_ambiguous_zygosity': 1,
  'condition_unparseable': 1, 'media_absent_or_unknown': 2, 'not_growth': 1,
  'malformed_header': 1}`, plus one `drop condition 'heat [37 C]'` line for column 4.
- Pure helpers: `_column_meta` (5 fields, 4 fields, malformed), `_readout_method`,
  `_parse_media` (rich vs synthetic, `+` base split, colony/spot/killing -> solid, LB -> None),
  `_parse_condition` (all nine `_UNIT_MAP` spellings, defensin -> peptide biologic, six
  unencodable shapes -> None as one exact list).

### Findings (pinned as the code behaves)

- `quinine` is absent from the compound table: its `Compound` has no identifiers and carries an
  `inchikey` gap with reason `deferred_pending_source_review`. The mirror test asserts on
  `compound.name` only, so this had not been visible. Whether quinine is the condition of a real
  pinned screen (the khozoie_avery_2009 entry, PMID 19416971, was the suspected one) is not
  verified from the repo; the real screen files are not read here.
- `non_orf_row` leads the drop Counter because rows of a kept column are walked before later
  columns are examined; the Counter order is insertion order, not the grammar order.
- `_parse_media` decides `state` from the phenotype string alone, so `SD` with `growth (killing
  zone)` is solid and `SC + EtOH` with `growth (pooled barseq)` is liquid; the `+ EtOH` supplement
  is discarded, not recorded.
