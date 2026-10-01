---
id: 3wqysfhenb49tevqmsifhfr
title: Auesukaree2009
desc: ''
updated: 1789272452083
created: 1789272452083
---

## 2026.09.12 - Serve-50 pass: shared YPD, the right comparator, two adjudicated genes

Rebuilt `EnvChemgenAuesukaree2009Dataset`
(`torchcell/datasets/scerevisiae/auesukaree2009.py`). The July build failed L0 on 100% of
its 525 records (it predates `Media.is_synthetic`), so a rebuild was mandatory; the content
fixes below rode along.

### What changed

- **Medium.** `Media(name="YPD", state="solid", is_synthetic=False)` with zero components
  is a node nothing joins. The paper states its own recipe, *"YPD medium (1% yeast extract,
  2% peptone, 2% glucose)"*, which is component-for-component the shared
  `torchcell.datamodels.media.YPD`, so the library object is now used verbatim.
- **The reference was wrong on two axes.** The paper's scoring definition is WITHIN a
  strain: *"Deletion mutants showing significantly reduced growth on plates under stress
  conditions, as compared to those under non-stress conditions, were defined as sensitive
  mutants."* The old loader copied the STRESSED environment as the reference and its
  `units` string claimed the comparator was the parental BY4742. The reference environment
  is now the UNPERTURBED base environment (shared YPD, 30 C, no perturbation, 72 h) -- 30 C
  for the heat records too -- and `MEASUREMENT_UNITS` states the within-strain comparator.
  `ReferenceGenome` cannot express "the same deletion strain", so `genome_reference` stays
  BY4742 and the comparator lives in `units`.
- **Typed category.** `category` was free text. It is now
  `ResponseCategory.sensitive` with `category_label="sensitive"`, and the reference is
  `ResponseCategory.no_change` with `category_label="tolerant"`. `sensitive` is deliberately
  NOT mapped to `reduced`: the paper reports an UNGRADED hit call, and a severity grade
  would be invented.
- **`assay_type=AssayType.spot_dilution`**, sourced by *"the serial-dilution spot test"*.
- **Every metadata number is a `SourcedValue`** quoting the mirrored `paper.md`:
  `_BACKGROUND`, `_MEDIUM`, `_SCREEN_CONDITIONS`, `_DOSES`, `_N_SAMPLES`, `_ASSAY`,
  `_SCORING`. Two readings are recorded as notes, not as quotes (see open flags).

### The two ambiguity adjudications

Running all 333 listed tokens through the shared `resolve_gene_name`: 283 RENAMED,
48 CURRENT, 2 AMBIGUOUS, 0 unresolvable. The old loader took `candidates[0]` silently.
`AMBIGUOUS` is now a hard stop, and the two are resolved by source evidence in
`_AMBIGUOUS_ADJUDICATIONS`; an unlisted ambiguous token raises.

- **PPA1 -> YHR026W (VMA16), not YBR011C.** The old mapping was measurably WRONG. YBR011C is
  IPP1, an essential gene: Costanzo 2021's Data File S1 carries it only as a
  temperature-sensitive strain (`YBR011C, IPP1, ipp1-5001, tsa1062`), so it cannot be in the
  4,828-strain NONESSENTIAL haploid collection this paper screened. YHR026W is VMA16/PPA1,
  the V-ATPase V0 subunit c'', and the paper lists PPA1 under "Vacuolar function" in both
  tables it appears in (Table 3 1-propanol, Table 4 heat). 2 records.
- **FEN1 -> YCR034W (ELO2), not YKL113C.** Both candidates are nonessential, so essentiality
  does not decide. The decisive evidence is that the SAME table lists the other candidate
  separately under its own name: Table 2 (methanol) has `Vacuolar function (11) CHC1, FEN1,
  VAM3, ...` and `Cell rescue, defense and virulence (5) AFT2, FLC1, FYV6, RAD27, SOD1`, and
  RAD27 IS YKL113C. 1 record.

Because an ambiguous token resolves to no single gene, it cannot round-trip through the
verifier's canonical-name rule, so these two records store the genome's canonical common
name (`VMA16`, `ELO2`) in `perturbed_gene_name`. Every other record keeps the paper's own
token verbatim. Neither re-keying collides: VMA16 and ELO2 appear nowhere in the tables.

### Drop rule and counts

Rule: a token is kept only when `resolve_gene_name` returns CURRENT or RENAMED; AMBIGUOUS is
adjudicated or raises. Measured over this release: **0 tokens dropped**, expected count
stays **525** (ethanol 95, methanol 55, 1-propanol 125, heat 178, NaCl 42, H2O2 30). The
accounting, including the two adjudications, is written to `dropped_records.json` beside
`processed/`.

### Methanol 54 vs 55

The abstract says 54 methanol-sensitive mutants; Table 2 lists 55 distinct genes and its
functional-class parentheticals also sum to 55 (11+7+4+5+2+5+2+2+4+5+1+7). The Table is the
per-gene DATA and is authoritative over the summary headline, so 55 records are stored. This
is a paper-internal count error, unchanged from the previous build.

### Provenance chain

- The article Tables 1-6 ARE the data; there is no supplementary deposit. They are extracted
  from the born-digital text layer of `paper.pdf` (sha256
  `01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1`) with
  `pdftotext -layout`, using each class row's parenthetical count as a self-checksum.
- `paper.pdf` is deposited at
  `$DATA_ROOT/torchcell-raw/auesukareeGenomewideIdentificationGenes2009/paper/paper.pdf`
  with a `manifest.json` recording `RetrievalMethod.zotero_attachment` (library 6582362,
  item IGDTEZJV, attachment VIJCFVIA; PMC file downloads use a JS proof-of-work and are not
  scriptable) and a `ProcessingRecord` carrying the `pdftotext -v` version string.
- Prose quotes anchor to the library mirror's `paper.md`, sha256
  `d0f3885d1f5027fc29beab7a4327ff377d2bc9c42dd5b87f580049eeda4223b2`. The MinerU OCR is
  LOSSY for the tables (its heat Table 4 "Unknown function" cell is truncated by YOR364w and
  YPL144w), which is why the PDF text layer is the source of record for the gene lists while
  `paper.md` anchors the prose. Every quote in the loader was verified to be an exact
  substring of `paper.md`.

### Verifier result (verbatim, 2026.09.12)

```
env_chemgen_auesukaree2009: PASS
  [ok] L0 structural: 525 records validated
  [ok] L1 count: observed 525, expected 525
  [ok] L1 pair_uniqueness: 525 unique (study, strain, condition) records, one each
  [ok] L1 canonical_gene_names: 333 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 uncertainty_sanity: 0 labelled uncertainties, none a zero dispersion; 525 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: categorical
  [ok] L3 reference_zero: categorical rule: all 525 references carry the baseline category 'no_change', which no experiment record reports
  [ok] L3 compound_identity: 347 compound references carry a structure identifier
  [ok] L3 media_membership: 525 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 333 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 333 measured systematic names is a gene of the current genome
```

### Open flags

- **`percent_v/v` is a convention, not a quote.** The paper writes "10% ethanol" and never
  writes "v/v" or "w/v" anywhere (0 hits over the full text layer). The unit is the standard
  alcohol-stress reading and is recorded in `_DOSES.note`.
- **`sample_unit=biological_replicate` is an interpretation.** *"All experiments ... were
  performed in triplicate"* does not say biological or technical; each replicate is an
  independent screening experiment of the whole collection, which is the biological reading.
  Recorded in `_N_SAMPLES.note`.
- **`media.YPD` is `state="solid"` but carries no agar component**, even though its own
  Tong and Boone quote names "20 g bacto agar". Adding one changes `media.YPD`'s node hash
  and so every dataset that uses it; it is safe today (only unserved datasets import it) but
  must be decided BEFORE any of them is admitted.
- **This dataset does NOT join the served YPD datasets at the media-node level.** Served
  loaders such as `ohya2005.py` emit their own inline `Media(name="YPD", state="liquid")`,
  a different node, and re-keying a served dataset is not something an increment can do.

### Files

- `torchcell/datasets/scerevisiae/auesukaree2009.py`
- `torchcell/adapters/auesukaree2009_adapter.py`, `torchcell/adapters/conf/auesukaree2009_adapter.yaml`
- `tests/torchcell/datasets/scerevisiae/test_auesukaree2009.py`, `tests/torchcell/adapters/test_auesukaree2009_adapter.py`

## 2026.09.12 - Medium switched from the YPD join anchor to the YPD_AGAR plate

`media.YPD` is the join anchor the family shares and is not used by a loader directly; a plate is `media.YPD_AGAR` (it states its agar row) and a culture is `media.YPD_LIQUID`. This screen is stamped onto plates, so both environments now carry `YPD_AGAR`, which puts these records on the SAME media node as Mota 2024's plates; the dev store was rebuilt (525 records, unchanged) and the verifier still reports `env_chemgen_auesukaree2009: PASS` with `L3 media_membership: 525 records on a shared MEDIA_LIBRARY medium (1 distinct media)`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: yes (mkdir before hash).

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_PDF_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Per-class count check and an in-table collapse ledger

Previous behavior: `_parse_tables` used each functional class's declared (N) only to decide whether an indented line continues the class, so a class declaring (5) and listing two genes parsed without error; only the per-stress total was checked. `_resolve_stress` collapsed a second token onto an ORF already claimed in the same table (`resolved.setdefault`) with no ledger entry, so `n_listed_tokens` could exceed `n_kept_records + n_dropped_records`.

Fix (issue #520): `_check_class_count` refuses a class whose parsed count differs from its declared count (checked when the next class row opens and when the table ends), naming the stress, the class label and both counts. A collapsed token is recorded as a `CollapsedToken` (stress, token, systematic name, kept gene name) in `DropLog.collapsed_tokens`, with `DropLog.n_collapsed_tokens`, so listed = kept + dropped + collapsed.

Record-neutral, measured on the pinned PDF (sha256 `01b94544...`): all 67 classes of Tables 1 to 6 match their declared count (the count gate still ends the NaCl table at a page-header line after its last class is complete); 0 repeated tokens within a table; the dev build ledger reads 525 listed, 525 kept, 0 dropped, and its LMDB holds 525 entries, so 0 collapses. The fixed parser returns the expected per-stress counts on the real PDF. `dropped_records.json` gains two keys (`n_collapsed_tokens: 0`, `collapsed_tokens: []`) on the next rebuild; records are unchanged.

Tests: `test_a_class_short_of_its_declared_count_refuses_at_table_end`, `test_a_class_over_its_declared_count_refuses_at_the_next_class_row`, `test_drop_log_ledgers_the_in_table_collapse`.

Review follow-up (same day): the "stress table not found" refusal and every `_resolve_stress` / `_resolve_token` refusal sat inside the write transaction, so a refusal left an empty `processed/lmdb` that a retry served. All six tables now resolve before the store is opened. On the synthetic layout build, the LMDB (8 records) is byte-identical to main's: digest `4505a1f3...`.
