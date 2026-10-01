---
id: pgqv7hyrvme86y6xakx4iki
title: Test_messner2023
desc: ''
updated: 1790482270066
created: 1790482270066
---

## 2026.09.26 - Hermetic build tests for the Messner 2023 KO proteome loader

Test file: `tests/torchcell/datasets/scerevisiae/test_messner2023.py` (12 tests, 17 cases with
the parametrized filename parser), covering [[torchcell.datasets.scerevisiae.messner2023]].
Part of the test-coverage campaign ([[plan.test-suite-buildout.2026.09.25]]); the module sat
at 26.6% line coverage before this file.

### Approach

`DATA_ROOT` is pointed at `tmp_path` and a five-line fake SGD GFF is written under
`data/sgd/genome/S288C_R64-5-1/saccharomyces_cerevisiae_R64-5-1.gff`, so
`build_uniprot_to_orf_map()` resolves without the real mirror. The matrix
(`yeast5k_noimpute_wide.csv`) and metadata (`yeast5k_metadata.csv`) are hand-written under
`<root>/raw/`; blank matrix cells stand for not-measured proteins.

### Fixture

Matrix (rows = UniProt accession, columns = sample filenames):

| Protein.Group | wt_a | wt_b | 10_9_hpr1_ko_YAL059W_ECM1_0.47 | 3_4_ko_YML009c_yml009c_0.9 | qc_1 | bad_ko |
|---|---|---|---|---|---|---|
| P00001 -> YBR001C | 10 | 12 | 8 | | 1 | 1 |
| P00002 -> YCR002W | 4 | | | 5.0 | 1 | 1 |
| P00410 -> Q0250 | 2 | 2 | 3 | | 1 | 1 |

Metadata: `wt_a`/`wt_b` are `HIS3` (WT, ORF YOR202W); the two `ko` samples delete YAL059W and
`YML009c` (lowercase in metadata); `qc_1` is `qc`; `bad_ko` is a `ko` sample with ORF
`YOR202W-not`; `absent_ko` is a `ko` row with no matrix column.

WT reference (mean, SD / sqrt(n), n over non-blank WT cells): YBR001C 11.0 / 1.0 / 2;
YCR002W 4.0 / NaN / 1; Q0250 2.0 / 0.0 / 2.

### Expected values asserted

- `len == 2` (`bad_ko` skipped by the nuclear-ORF regex, `absent_ko` ignored, `qc_1` ignored).
- `data.csv == "filename,orf,gene,n_proteins\n<KO_A>,YAL059W,ECM1,2\n<KO_B>,YML009C,yml009c,1\n"`;
  `gene_set.json == ["YAL059W", "YML009C"]`.
- Record 0 (YAL059W/ECM1): `protein_abundance {YBR001C 8.0, Q0250 3.0}`, `n_replicates` all 1,
  `protein_abundance_se None`, `Environment(media=SM, temperature=30)`, BY4741; reference
  restricted to the two measured proteins with `{11.0, 2.0}`, SE `{1.0, 0.0}`, n `{2, 2}`;
  publication PMID 37080200 / DOI 10.1016/j.cell.2023.03.026. Whole-record `model_dump()`
  equality, exact.
- Record 1 (`YML009c` -> `YML009C`, gene `yml009c`): abundance `{YCR002W 5.0}`; reference
  `{YCR002W 4.0}`, n 1, SE NaN.
- Two reference-index entries `[[0], [1]]` because the two KOs measured different protein sets.
- `build_uniprot_to_orf_map` returns exactly `{P00001: YBR001C, P00002: YCR002W, P00410: Q0250}`;
  `ID=YCR002W_mRNA` still yields `YCR002W`; a second line with a seen accession does not
  overwrite; a seven-column line mentioning `UniProtKB:P99999` is skipped; no GFF ->
  `FileNotFoundError`.
- Error paths: unmapped accession -> `RuntimeError("1 Messner proteins have no UniProt->ORF
  mapping (e.g. ['P99999'])")`; no HIS3 columns -> `RuntimeError`; no GFF -> `FileNotFoundError`
  before metadata is read.
- Manifest: `dataset_name proteome_messner2023`, loader class/module, hostname, and a closure
  containing the five directly imported schema symbols.

### Findings (pinned as the code behaves)

- `_gene_from_filename` returns whatever token follows the ORF token without validating it:
  `3_4_ko_YAL059W_0.47` yields the gene name `"0.47"`. Real filenames carry a gene token
  between the ORF and the numeric suffix, so this only bites on a malformed filename.
- A protein measured in one WT sample gets `float("nan")` as its reference SE (not None), and
  the reference dict still carries the key.

## 2026.09.30 - Phase 14: numeric gene names, linear values, the GFF map, downloads

Twelve to twenty-four tests, 81 to 100 percent. The real filename shape `10_9_hpr57_ko_YBL007C_2824_0.49` stores `perturbed_gene_name` "2824" beside a second strain named "SLA1" (issue #485, whole record); values are linear (WT samples 1 and 1001 give a reference of 501.0 with SE 500.0; 393221 and 0.0313 stored verbatim); `duration_hours` None (issue #486); a protein a KO measured but no WT sample did raises a bare `KeyError('YBR001C')` (line 360); the exact build-summary log lines; a GFF line with no ORF token skipped and a line with two accessions mapping both; `download` when the mirror file is missing, off the pin, and when both files copy; a raw file already present kept without hashing (190); `main` and the schema classes.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_download_skips_a_present_raw_file_without_hashing_it` (#528) is now `test_a_present_raw_file_off_the_pin_is_refused_at_build_time`. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
