---
id: cljz5kturlsnt8neeib5r3k
title: Natural_isolate
desc: ''
updated: 1791808458110
created: 1791808458110
---

## 2026.10.12 - One genotype per isolate of the Peter 2018 panel (#988, #987, #1001)

`torchcell/sequence/natural_isolate.py` builds the genotype of every natural isolate of the Peter 2018 1,011-genome panel, for every loader whose records are such isolates: Caudal 2024 (`caudal2024.py`), Peter 2018 (`peter2018.py`), Muenzner 2024 (`muenzner2024.py`, both stores) and De Chiara 2022 (`dechiara2022.py`, all three stores). Before this module the construction lived in the Caudal loader, and the other three imported its private helpers (`_parse_fasta`, `_reference_slice`, `_isolate_and_symbol`, `_demangle_orf`, `_orf_to_s288c`, ...) and called its static methods (`_content_perturbations`, `_sequence_perturbations`, and on Muenzner `_sequence_variants` with its own dataset cast to Caudal's). Peter and De Chiara each re-implemented the tarball diff. Issues #988 and #1002 asked for one module.

### Where it lives, and why not `peter2018_genotype.py`

The issue named two paths. The loaders' notes point at the first one through #1002 (`torchcell/datasets/scerevisiae/peter2018_genotype.py`), and that path is refused by the raw-pin gate: part B of `tests/torchcell/datasets/scerevisiae/test_raw_pins.py` (issue #561) refuses any module in `torchcell/datasets/scerevisiae/` that carries a sha256-named identifier or a 64-hex constant without calling `verify_raw_files` in a `process()`, and a shared builder has no `process()`. Weakening that gate to make room was not an option. The second path, `torchcell/sequence/natural_isolate.py`, sits beside the genomes-tier registry the six input files come from.

### What a genotype carries

| element | source | class |
|---|---|---|
| allele differs from the S288C R64 slice | `allReferenceGenesWithSNPsAndIndelsInferred.tar.gz` | `SequenceVariantPerturbation` |
| non-reference ORF present (copy number) | `genesMatrix_PresenceAbsence.tab.gz`, `genesMatrix_CopyNumber.tab.gz` | `NaturalGenePresencePerturbation` |
| reference ORF absent | presence matrix | `NaturalGeneAbsencePerturbation` |
| HO deleted (#988) | Supplementary Table 1, 'HO deletion' = yes | `DeletionPerturbation` of YDL227C, beside whatever the presence matrix states about HO |
| a released loss-of-function call (#987), keyed as its matrix keys the row | `1011LossOfFunction.xls.gz`, `genesMatrix_Frameshift.tab.gz` | `NaturalLossOfFunctionPerturbation` |
| ploidy, zygosity, aneuploidy, manipulation (#1001) | Supplementary Table 1 | `Genotype.natural_isolate` (`NaturalIsolateDescriptor`) |

The Supplementary Tables workbook (`41586_2018_30_MOESM3_ESM.xls`, sha256 `b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273`, the library mirror's `si/si3.xls`) was registered into the genomes-tier set `peter2018_1011_assemblies` on 2026-10-12 by `register_supplementary_tables`, with the `pmc_cloud` retrieval record the library manifest holds (`https://pmc-oa-opendata.s3.amazonaws.com/PMC6784862.1/41586_2018_30_MOESM3_ESM.xls`). Every loader therefore reads all six genotype files (`GENOTYPE_FILES`) from one tier, each verified against its manifest.

### Decisions

**The data is the data (owner, 2026-10-12).** Everything Peter 2018 releases about an isolate is stored as released, keyed as the release keys it. Nothing is looked up in another release file or in the S288C annotation at load time to admit, drop, re-key or annotate a record. A first version of this module did all three (an allele status read from the tarball, a presence cell read from the presence matrix, code-less frameshift rows named through the R64-4-1 GFF with two rows refused, and an HO deletion that replaced the matrix's natural absence); the owner's correction removed them, and the decisions below are the result. The construction the loaders already shared (sequence variants diffed from the release's inferred alleles, present pangenome ORFs, natural absences) is the release's own representation off the 1,011-genome set and is unchanged.

- **HO deletion.** Paper.md line 174 (sha256 `21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da`): "However, among the 918 selected isolates, 124 were non-natural haploid with the $H O$ gene deleted and the 93 external isolates were genetically manipulated and made homozygous before sequencing." Line 36: "As 217 strains were genetically manipulated and no longer in their natural ploidy states, we assigned a relative ploidy state to the other 794 isolates (Methods)". Table S1's 'HO deletion' column reads yes for 124 isolates (122 of the 971 phenotyped). The deletion is stored as the parent `DeletionPerturbation`, because every child class asserts a cassette (KanMX, NatMX, a named marker) that neither the table nor the paper names; it was an abstract parent and is now a union member, fields unchanged. Where the presence matrix also marks HO absent, both released statements are stored: the genotype carries the deletion and the natural absence.
- **The other 93 manipulated isolates.** Table S1 does not mark which isolates are the 93 external ones. Measured, not used: exactly 93 Table S1 rows carry a `SACE_` standardized name, all cite Table S1 references 2, 6 or 17 (Strope 2015, Bergström 2014, Skelly 2013, the paper's refs 8, 7 and 6) and all are 2n homozygous. Because that identification is an inference from naming, every isolate whose 'HO deletion' reads no carries a typed gap on `genetic_manipulation` quoting the paper, rather than a value.
- **Loss-of-function cause.** One member per release matrix, named as the release describes it. The host's `README.txt` (deposited in the raw mirror as `data/README.txt`, sha256 `f48e32771b44d38e88b9c8833f6f47e1a30eb6634ffccf55be42b117dd24f025`, by `register_release_readme`; line 14): "1011LossOfFunction.xls.gz: loss of function predicted for all genes in the 1011 dataset (nonsense mutation + SIFT predictions)". The release folds the two predictions into one bit, so the member is `nonsense_or_sift`. `homozygous_frameshift` is the frameshift matrix (Data availability, paper.md line 290).
- **Every released call is stored, keyed as its matrix keys the row.** A call carries its matrix (`call_ref`) and the row's own keys: `release_code` (`CODE`), `release_name` (the loss-of-function matrix's `NAME`; the frameshift matrix has no such column), `release_gene_name` (`Gene`) and `release_span` (`CH`/`START`/`STOP`, R64-1-1). Its gene key (`systematic_gene_name`, also `perturbed_gene_name`) is `NAME` for a loss-of-function row, the pangenome ORF name the release gives the code for a coded frameshift row (the presence-matrix column `X<code>.<ORF>`, the name the natural absences carry), and `Gene` verbatim for a frameshift row released without a code. On the 533 code-less rows, `Gene` is a standard name on some (PAU8, SLM6, AAD16, DDI3, ...), so the yeast systematic-name pattern is relaxed on this leaf, as it already is on `NaturalGenePresencePerturbation`. Over the 971 phenotyped isolates: 231,339 loss-of-function calls on 4,069 genes and 141,453 frameshift calls, 8,625 of them code-less.
- **No call is annotated against another file.** Whether the inferred-allele release has an allele for the same gene, whether that allele equals S288C, and whether the presence matrix marks the ORF absent are not stored on a call: they are cross-references the release does not make. Those disagreements were measured for #987 by `experiments/036-dataset-fixes-before-kg-build/scripts/peter2018_release_inventory.py` and stay there.
- **The variant cache** (`preprocess/sequence_variants.parquet`) now has a sidecar (`sequence_variants.index.json`) naming the tarball's sha256, the isolates diffed and the tarball's genes; it is reused for any subset of those isolates at that sha256, so a verifier re-derives the stored isolates from the build's own cache.

### Verification shared by the four loaders

`torchcell/verification/natural_isolate_genotype.py` (`IsolateGenotypeAudit`): every distinct stored genotype is fed once; L1 `isolate_genotype_identity`, L2 `isolate_genotype_round_trip` (dump equality with the builder's re-derivation), L4 `isolate_descriptor_matches_table_s1` (re-read from the library mirror's `si/si3.xls`, not the tier copy) and L4 `loss_of_function_calls_match_release` (the stored calls per isolate and cause against the matrices' 1s, re-read with pandas). Caudal gained a `verify_build` so `build_dataset_lmdb --verify` covers it (`runners.verify_rnaseq_store`, the RNA-seq family rules plus the audit).
