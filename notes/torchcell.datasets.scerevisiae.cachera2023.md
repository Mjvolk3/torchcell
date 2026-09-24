---
id: sfk463pvqf0q3yk8urthewq
title: Cachera2023
desc: ''
updated: 1783223952046
created: 1783223952046
---

## 2026.07.05 - WS8 build: Cachera CRI-SPA betaxanthin

Roadmap [[plan.schematization-ingestion-roadmap.2026.06.23]] WS8. The abstract's
betaxanthin case. First `MetabolitePhenotype` dataset (WS4).

### What the data is (from OCR'd paper.md)

Cachera 2023 (NAR, `10.1093/nar/gkad656`) is a METHOD paper (CRI-SPA). It transfers
**four betaxanthin-biosynthesis genes** into each strain of the ~4800-strain YKO
collection and reads out per-colony **betaxanthin** (a yellow, naturally fluorescent
plant metabolite) by image analysis. The "CRI-SPA score" is a corrected/normalized
colony fluorescence intensity -- a QUANTITATIVE proxy for betaxanthin level. Because it
is population-centered it can be negative. So this is a `MetabolitePhenotype`
(continuous), NOT the ordinal VisualScore of Ozaydin -- the data shape picked the schema.

### Source (data is on GitHub, not the PDF)

The paper's **Data Availability** points to the CRI-SPA GitHub repo. Zotero holds only
the PDF (OCR'd → paper.md). The genome-wide per-gene data lives at
`github.com/pc2912/CRI-SPA_repo` (the OCR dropped the `_repo` suffix). We ingest
`GA1_2_4_6.csv` (gene-level corrected+filtered, replicates 1/2/4/6, 4788 rows) using
the 24 h `corrected_mean_intensity` (mean/std/count) → betaxanthin level + SE + n. All
mirrored under `torchcell-library/cacheraCRISPAHighthroughputMethod2023/` with a
sha256 `manifest.json` (paper.pdf, paper.md, si/GA1_2_4_6.csv).

### Dataset (`BetaxanthinCachera2023Dataset`)

Source gene names are COMMON (AAC1), so an injected `SCerevisiaeGenome` resolves them
to systematic ORFs (same pattern as Sameith). **4735 records** (one per ORF). Excluded,
all logged: 28 control/NaN rows (incl. 'WT'), 5 unresolved common names, 20
common-name→same-ORF collisions (deduped, keep first). L0-L4 all PASS
(`torchcell/verification/metabolite.py`); L4 gene overlap with the Ohya deletion
collection = **0.982**.

### Schema (WS4) `MetabolitePhenotype`

`metabolite_level: dict[metabolite_id -> float]` (Yeast9 `s_NNNN` where native, or a
product name for heterologous betaxanthin), `metabolite_level_se`, `n_replicates`
(per metabolite), `measurement_type` (what the number IS -- here
`cri_spa_corrected_fluorescence_intensity_24h`, so assays are never silently mixed),
optional `target_metabolite_ids` for Yeast9/CBM linkage (None -- deferred).

## 2026.07.05 - Follow-up Fixes + Cassette Provenance + Perturbation Design

### Done + committed (branch `fix/ws8-ozaydin-cachera-followups`)

- **PubMed ID.** `Publication` now carries `pubmed_id="37572348"` + `pubmed_url`.
  PMID<->DOI (`10.1093/nar/gkad656`) confirmed via NCBI E-utilities (title "CRI-SPA:
  a high-throughput method...", Nucleic Acids Research 2023). Not guessed.
- Cachera has no `qc_flags` field (that was Ozaydin's `VisualScorePhenotype`); no
  change to `MetabolitePhenotype` here. The canonical LMDB still has the OLD
  `pubmed_id=None` publication and should be rebuilt in place when this lands (needs
  the injected `SCerevisiaeGenome` to map common gene names).

### Btx-cassette definition -- sourced from OUR mirror (provenance-first)

From `torchcell-library/cacheraCRISPAHighthroughputMethod2023/paper.md` (MinerU OCR):

- **Btx-cassette** (paper calls it a "five-gene" cassette; the abstract's "four
  genes that enable betaxanthin production" = the 4 non-marker genes):
  - Heterologous plant genes: **CYP76AD1** (cytochrome P450) + **DOD** (DOPA
    4,5-dioxygenase) -- the two genes strictly required for betaxanthin (ref 25).
  - Native yeast genes, feedback-resistant mutant alleles: **ARO4^K229L**
    (DAHP synthase, YBR249C) + **ARO7^G141S** (chorismate mutase, YPR060C) -- relieve
    shikimate-pathway negative feedback.
  - Selectable marker: **natMX** (not a betaxanthin gene).
- **Localization = chromosomal_integration** at expression site **XII-5** (BY-Btx
  made by NotI-excised insert of **pBTX2** into BY4741; CD-Btx from pBTX1 targets
  XII-5). Contrast Ozaydin's episomal 2micron plasmid.
- **Full sequence (#4, external):** plasmids **pBTX1 / pBTX2** in Supplementary
  Table S2 and the CRI-SPA repo **github.com/pc2912/CRI-SPA_repo** (also the source
  of the ingested `GA1_2_4_6.csv`). Our mirror has the composition + locus; the
  plasmid sequences are the remaining external dig.

### Heterologous-cassette perturbation -- DESIGN ADOPTED (see 2026.07.15 below)

Same schema gap + three-option decision as documented in
`[[torchcell.datasets.scerevisiae.ozaydin2013]]` (recommendation = per-gene
`GeneAdditionPerturbation` with a `localization` field). Cachera exercises the
`chromosomal_integration` localization + a `locus="XII-5"` value, and the native
feedback-resistant alleles (ARO4^K229L / ARO7^G141S) are the concrete case for the
"native extras" sub-decision (new addition type flagged native vs reuse
`AllelePerturbation`). **Resolved (2026.07.15):** recommendation A is implemented in
`cachera2023.py::_betaxanthin_cassette` -- see the dated section below.

### 2026.07.05 - Plasmid availability (answer: YES, GenBank maps provided)

Verified by web research (do they provide the plasmid so it can be downloaded?):

- **Data Availability (paper, verbatim):** "All plasmid maps are available for
  download as GenBank files." So pBTX1 / pBTX2 (+ the CRI-SPA vectors) DO exist as
  downloadable GenBank maps -- the raw artifact the "plasmid-seq + feature
  annotation -> collapse to GeneAddition" path needs.
- **Location:** NOT in the GitHub repo `github.com/pc2912/CRI-SPA_repo` (that has
  only data CSVs + analysis notebooks, no `.gb/.fasta/.dna`). The maps are in the
  **OUP supplementary archive**:
  `https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar/51/17/10.1093_nar_gkad656/1/gkad656_supplemental_files.zip`
  (contains Suppl. Tables S1 strains / S2 plasmids / S3 primers + figures/methods).
- **Retrieval method = manual_browser (un-scriptable).** The silverchair CDN URL is
  a CloudFront SIGNED url: bare `curl` returns HTTP 403 `MissingKey` (needs a
  Key-Pair-Id token from an academic.oup.com browser session). So this is the
  nature.com-class case in our provenance notes: manual-once download -> deposit to
  `torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/` + sha256 -> then
  reproducible via our mirror. No Addgene ID / GenBank accession is given for pBTX1/2.
- Upstream heterologous parts ARE on Addgene (e.g. `pL0-BvCYP76AD1` #162529, a MoClo
  L0 part; DOD from the DeLoache 2015 betaxanthin biosensor lineage), useful as
  cross-checks but they are PARTS, not the assembled pBTX1/pBTX2.

## 2026.07.15 - Pre-adapter audit: gene-drop documentation + design adopted

Pre-adapter cleanup ahead of the BioCypher/graph-DB rebuild
([[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]). The loader code is current and
rebuilds clean: **4735 usable ORFs** (matches the paper's ~4800-strain / 4761-gene
genome-wide CRI-SPA screen; the data is the authors' `GA1_2_4_6.csv`, not a PDF selection).

### Unresolved gene names -- 5 dropped, by design (NOT backfilled)

`process()` resolves source common names to systematic ORF ids via the injected
`SCerevisiaeGenome`; 5 names do not resolve and are dropped (never guessed into an ORF):

- `WT` -- wild-type control row, correctly excluded.
- `YLR287-A` -- malformed systematic id (missing the trailing W/C), excluded.
- **`AAD6`, `CRS5`, `FLO8`** -- real *S. cerevisiae* genes (Aad6 aryl-alcohol
  dehydrogenase; Crs5 copper metallothionein; Flo8 flocculation TF). Their SGD systematic
  ids **YFL056C / YOR031W / YER109C are absent from the reference `gene_set` (6607)** used
  to build the cell graph, so admitting these 3 records would create perturbations pointing
  at gene nodes that do not exist in the KG (orphaned references).

**Decision: document, do NOT backfill.** Injecting an alias map to force these in would
(a) require guessing/sourcing systematic ids (against the no-guess rule) and (b) orphan
gene nodes downstream. Dropping is the graph-safe, provenance-correct behavior. Net loss
is **3 records / 4735** (0.06%). The `process()` log now prints the full unresolved list
(not a truncated sample) so the drop is auditable. **Follow-up:** revisit if the reference
genome annotation is updated to include YFL056C/YOR031W/YER109C, at which point these 3
strains can be admitted with valid gene nodes. This same alias-map behavior affects other
common-name loaders (Sameith, caudal2024) and may warrant a central fix.

Additionally, **20 ORF collisions are deduped (first-kept)**: distinct source names that
alias to the same systematic ORF keep the first occurrence; the rest are logged and
skipped (already existing behavior, called out here for the rebuild audit).

### Cassette-perturbation design -- ADOPTED (recommendation A)

The three-option decision (see the PENDING section above, now resolved) is settled and
**implemented in code**: each cassette gene is a per-gene `GeneAdditionPerturbation` in the
genotype's `perturbations` list with `localization="chromosomal_integration"`,
`integration_locus="XII-5"`, `source_organism`, `construct_name="Btx-cassette"`. The native
feedback-resistant alleles **ARO4^K229L (YBR249C) / ARO7^G141S (YPR060C)** are carried as
`variant="K229L"` / `"G141S"` on `GeneAdditionPerturbation` -- NOT as `AllelePerturbation` --
because they are integrated ectopically at XII-5, not edited at the native locus. So the
Cachera genotype is `{kanmx_deletion: 1, gene_addition: 4}` (deletion + CYP76AD1 + DOD +
ARO4^K229L + ARO7^G141S; the natMX marker is omitted as a non-betaxanthin gene).

### Rebuild note

The on-disk canonical LMDB predated the required `Media.is_synthetic` field and failed
schema round-trip; it was rebuilt in place under `$DATA_ROOT` as part of this cleanup.
sha256 verification of `GA1_2_4_6.csv` (`DATA_SHA256`) was added to `download()`.

## 2026.07.15 - Resolve gene names via the shared genome resolver (retain pseudogenes)

`_resolve_systematic` now delegates to `SCerevisiaeGenome.resolve_gene_name` (see
[[torchcell.sequence.genome.scerevisiae.s288c]]) instead of the gene-only
`alias_to_systematic`. This closes the standing follow-up in this note: common names that
resolve to a valid non-`"gene"` R64 feature (AAD6/CRS5/FLO8 -> YFL056C/YOR031W/YER109C,
`blocked_reading_frame` pseudogenes) are now **retained** as real loci rather than dropped as
"unresolved". The resolver's standard-name layer also fixes common-name disambiguation (e.g.
AAP1 -> YHR047C, not the mito Q0080).

Outcome: **4647 -> 4719 usable ORFs**; unresolved drops fell 85 -> 11 (only the `WT` control,
the malformed `YLR287-A`, genuinely AMBIGUOUS common names FEN1/PPA1, and a few retired
dubious ORFs). Removed the local `_SYSTEMATIC_RE` regex path. The heterologous Btx-cassette
(CYP76AD1/DOD + ARO4/ARO7 variants) is unaffected (separate fixed constant).

## 2026.09.23 - Issue 195 resolution

Issue [#195](https://github.com/Mjvolk3/torchcell/issues/195) reported that the dev build
was missing nine genes (RIP1, PRS3, SDH1, APT1, TSL1, PFK2, NRK1, MSF1, ANT1) because the
LMDB predated the resolver landing in `567fa6aa` (2026-07-15 14:40). That symptom is gone,
and a second defect was found in its place.

### The nine genes were already present

The dev LMDB at `$DATA_ROOT/data/torchcell/betaxanthin_cachera2023/processed/lmdb` was
rebuilt 2026-09-14 02:27, owned by `michaelvolk` (`processed/pre_filter.pt` link count 1,
not the uid-7474 hardlink the issue observed), and holds 4,719 records. All nine genes are
present by systematic name: YEL024W, YHL011C, YKL148C, YML022W, YML100W, YMR205C, YNL129W,
YPR047W, YPR128C. A resolution census over the raw file with the current code reproduces
the count exactly, so the build was current: 4,788 raw rows, minus 28 control/NaN rows,
minus 11 unresolvable names (AMBIGUOUS FEN1/PPA1 plus 9 retired, including the `WT` control
and the malformed `YLR287-A`), minus 30 ORF collisions, leaves 4,719.

### The real defect: the varying deletion stored the ORF id twice

Every `kanmx_deletion` in the old build had `perturbed_gene_name == systematic_gene_name`
(4,719 of 4,719), while the fixed cassette carried standard names (`YBR249C` / `ARO4`). The
source column `gene` in `raw/GA1_2_4_6.csv` is common-named for most rows, and the loader
resolved it to an ORF and then discarded the name, so a record could not say what the paper
reported. Peer metabolite-screen loaders over the YKO collection do not do this: Lopez
stores `std_map.get(orf, orf)` (`lopez2024.py:279`), Xue stores the source common name
(`xue2025.py:322`), and the SGD essentiality loader stores `TFC3` for `YAL001C`.

Storing the systematic id in both fields is legal where a source supplies no common name
(Mulleder, Ozaydin, Sameith, Hillenmeyer, Yeastphenome all do it, and the L1
`canonical_gene_names` round-trip check in `torchcell/verification/common.py:507-622` passes
on it because an ORF id resolves back to itself). It is wrong here because the source DID
supply a name.

The name now comes from the GENOME, not from the source column, via
`smith2006.canonical_common_names` (the helper Smith 2006, Smith 2016, Mormino and Lian
already share). Its docstring is the convention: one spelling per gene, taken from the
genome so the spelling is identical across datasets, and only a standard name that resolves
BACK to the gene is used. Measured over the 4,719 retained rows:

| quantity | count |
|---|---|
| records whose `perturbed_gene_name` is now a common name | 3,930 |
| ORFs with no round-tripping standard name, id kept in both fields | 789 |
| rows whose 2023-era source spelling the genome has since superseded | 408 |

The 408 are the reason for preferring the genome: ACN9 is now SDH7 (YDR511W), AIM1 is now
BOL3 (YAL046C), ADE5,7 is now ADE57 (YGL234W). Every `(systematic_gene_name,
perturbed_gene_name)` pair stays unique, so the L1 ORF-uniqueness check cannot collapse the
way it did for Costanzo 2021 (`costanzo2021.py:709-716`); the loader already dedups by ORF
before writing, which is what guarantees it.

### Rebuild and verification

`processed/` and `preprocess/` were moved to
`/scratch/projects/torchcell-deprecated/2026.09.23/betaxanthin_cachera2023-{processed,preprocess}-pre-issue195`
and rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset
BetaxanthinCachera2023Dataset`: 4,719 records, gene_set 4,721, 1 reference, in 3 s. The
metabolite verifier (`torchcell.verification.runners.run_metabolite`) passes all eight L0-L4
checks on the new build, including L1 genotype uniqueness at 4,719 unique strains and L4
gene containment 0.990 against Ohya 2005; the report is at
`$DATA_ROOT/data/torchcell/betaxanthin_cachera2023/preprocess/verification_report.json`.

`test_cachera2023.py` had been asserting 4,735 records and was failing before any change in
this work, a stale expectation from before `567fa6aa`. It now asserts 4,719 and a second
test pins the naming convention.

### Admission: BLOCKED, so this waits for the next full KG build

`perturbed_gene_name` is hashed into the experiment, genotype and perturbation node ids
(`cell_adapter.py:527-528,540-542,595-598`), so changing it changes the content id of every
record that carries a common name. The check says so verbatim:

```
Admission check: BetaxanthinCachera2023Dataset  ->  BLOCKED
  served: yes; superset proof from bolt://localhost:7687: 4719 served ids, 4719 in the
  dev LMDB, 3930 served ids missing from it, 3930 to add
  [BLOCK] BetaxanthinCachera2023Dataset is already in the served store and its dev LMDB
  no longer produces 3930 of the 4719 served experiment ids (full rebuild required:
  incremental import would leave those nodes beside their replacements).
```

The served store holds 4,719 Cachera experiments, the same count the pre-change dev build
produced, so the served graph was not missing the nine genes either. The 789 records whose
two fields legitimately coincide are unchanged and account for the overlap. This keeps the
issue's `before-next-kg-build` label: the change is correct in the dev tree now and reaches
the graph on the next full rebuild, never by increment.

### Other pre-resolver builds: none

`python -m torchcell.provenance.build_manifest --data-root /scratch/projects/torchcell-scratch`
reports 77 built datasets, 51 fresh, 7 stale, 19 unmanifested. Sixteen loaders call
`resolve_gene_name`, and every one of their live dev builds is dated 2026-09-12 or later, so
no other dataset carries the gap this issue describes. The 7 stale are the four Costanzo
subsamples (`dmf|dmi_costanzo2016_{1e5,5e5}`, 2026-07-16) plus `env_chemgen_auesukaree2009`
and `env_chemgen_vanacloig2022`, and they are stale on the SCHEMA contract, not on the
resolver. Note the limit of that tool: it fingerprints the schema closure, so a build that
predates a RESOLVER change reads `fresh`. Build date against `567fa6aa` is the check for
that, and it is what was used here.

### Related item: the relative `genome_root` / `go_root` defaults

Half of the issue's footgun is already fixed. `SCerevisiaeGenome.__attrs_post_init__` now
resolves all four release files through the genomes-tier registry
(`resolve(self.ASSEMBLY_SET, ...)`, `s288c.py:504-523`), and the download path was deleted,
so a bare constructor no longer re-downloads a genome tree. See
[[plan.genomes-tier.2026.09.14]] decision 4.

What remains is real but is NOT a small safe fix, so it was left alone. `go_root` still
defaults to the relative `"data/go"` (`s288c.py:470`) and `s288c.py:556-561` downloads
`go.obo` into it with `download_url`, unpinned, which currently 403s; `genome_root` still
defaults to the relative `"data/sgd/genome"` (`s288c.py:469`) and `data.db` is still written
under it. Anchoring either default to `DATA_ROOT` would redirect every bare constructor (18
of them, all in `__main__` demo blocks such as `torchcell/datasets/go.py:166` and
`torchcell/graph/sgd.py:299`) onto the SHARED genome cache, and because `overwrite` defaults
to `True` (`s288c.py:471`) such a constructor calls `gffutils.create_db(force=True)` on the
shared `data.db` that live training jobs hold open. That is the rebuild race recorded in
[[plan.genomes-tier.2026.09.14]] gotcha 3, and the plan's decision 6 defers the `overwrite`
default to its own PR at 67 call sites. The honest order is therefore: flip `overwrite` to
default `False` with a DDP-safe build-when-absent path first, then anchor these two defaults
in the same PR. The `go.obo` download deserves the same treatment the release files got, a
sha256-pinned tier entry with no download path, rather than an anchored relative path.
