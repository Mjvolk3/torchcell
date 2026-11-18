---
id: q8rmfnyhhl80ts3fiq8to3s
title: P Putida Cgt Mvp Goal
desc: ''
updated: 1794968229407
created: 1794968229407
---

## 2026.11.17 - Scope, End Criterion, and What Is Measured So Far

This note is the standing reference for the *Pseudomonas putida* CGT + metabolic
module program. It is written to survive across sessions: the end criterion in
"MVP end criterion" is the contract, and every later session should read it
before adding work.

### Why P. putida, and what the phenotype is

The target is aviation biofuel. *P. putida* KT2440 carrying a heterologous
mevalonate route secretes isoprenol, the precursor to the sustainable aviation
fuel candidate. The phenotype we predict is isoprenol titer in mg/L at the 48 h
assay point, and the genotype is a multiplexed CRISPRi design: 1 to 4 sgRNAs
against native genes, carried on a dCas9 vector, in a strain that also carries
the pathway plasmid.

This is the bacterial counterpart of the yeast 031 program (CGT plus metabolic
module, betaxanthin as the metabolite readout). Here the metabolite readout is
isoprenol, and the metabolic model already contains the exchange reaction for
it, so the readout is a metabolite token in the same sense 046 uses one.

### Strain and genome: already in the tier, nothing to provision

The strain question is settled and the genome is already pinned. There is no
genome work in this program.

| item | value | where |
|---|---|---|
| organism, strain | *Pseudomonas putida* KT2440 | Dryad metadata `Organism`; `kt2440.py` |
| base engineered chassis | `IY1449b` (+ pIY670 = `IY1452b`) | Dryad `Strain_ID`; loader |
| assembly set id | `pputida_KT2440_ASM756v2` | `torchcell/sequence/genome/registry.py:108` |
| GenBank / RefSeq assembly | `GCA_000007565.2` / `GCF_000007565.2` | `KT2440_ASSEMBLY` in `kt2440.py` |
| replicon | `AE015451.2` (= NC_002947.4), 6,181,873 bp | `kt2440.py` |
| locus tags | `PP_####`, 5,786 loci (5,621 numbered) | `KT2440_LOCUS_TAG_PATTERN` |
| pinned bytes | 10 files + `manifest.json` + `SHA256SUMS`, `provenance_complete: true` | `$DATA_ROOT/torchcell-genomes/pputida_KT2440_ASM756v2/` |
| GO | `109.P_putida_KT2440.goa` (GAF 2.2, taxon 160488) | same |

`PPutidaKT2440Genome(BacterialGenome[PPutidaKT2440Gene])` is the exact analogue
of `SCerevisiaeGenome`, sitting on a 1,245-line shared bacterial layer
(`torchcell/sequence/genome/bacterial.py`) with GenBank-first ingest, a RefSeq
`old_locus_tag` crosswalk and a six-layer `resolve_gene_name`. Siblings already
exist for E. coli K-12, REL606, W3110 and *P. kudriavzevii* SD108.

Key files:

- `/home/michaelvolk/Documents/projects/torchcell/torchcell/sequence/genome/pputida/kt2440.py`
- `/home/michaelvolk/Documents/projects/torchcell/torchcell/sequence/genome/bacterial.py`
- `/home/michaelvolk/Documents/projects/torchcell/torchcell/sequence/genome/registry.py`
- `/home/michaelvolk/Documents/projects/torchcell/scripts/provision_bacterial_genomes.py`

### What already exists, and what the gaps actually are

This was the biggest correction of the first session. The P. putida side of
torchcell is not a blank slate: there are **16 loaders in
`torchcell/datasets/pputida/` (44,212 lines) and 32 served P. putida datasets in
KG 4.0.**

The Carruthers campaign itself is **already loaded and already served**:
`torchcell/datasets/pputida/carruthers2025.py`, 6,593 lines, the largest loader
in the package, with four dataset classes.

| class | experiment type | served records |
|---|---|---|
| `IsoprenolTiterCarruthers2025Dataset` | `ProductTiterExperiment` | 502 (465 campaign + 37 panel) |
| `CampaignProteomeCarruthers2025Dataset` | `ProteinAbundanceExperiment` | 465 |
| `ProteomeCarruthers2025Dataset` | `ProteinAbundanceExperiment` | 21 |
| `ProteomeFoldChangeCarruthers2025Dataset` | `ProteinFoldChangeExperiment` | 16 |

Record identity is (CRISPRi construct, DBTL cycle). The loader already models
everything I had assumed was owed: the `IY1449b` chassis on a `ReferenceGenome`
pinned to `pputida_KT2440_ASM756v2`, the five pIY670 pathway genes as
`HeterologousPathwayPerturbation`, each guide as a
`CrisprInterferencePerturbation` on its `PP_` tag, the 86 kb host deletion as a
`BackgroundAllele` / `AlleleEdit` / `GenomicSpan` of 86,812 bp at
4,538,575..4,625,386 (resolving `zwf`, `glZ`, `liuC`, `ldhA`), and
`M9_NREL_CARRUTHERS2025` as the environment. The served 465 reconciles with the
paper's 472 strains because 7 strains carry six replicates, which matches the
472 design-cycle cells measured above.

So the real gaps are these, and they are the work:

1. **The DBTL cycle label is not materialized in the served graph.** It is the
   loader's record identity and it sits in the raw mirror's `DBTL_Cycle`
   column, but a permissive regex over the full serialization of every record in
   all four Carruthers stores finds no DBTL token, and `screen_id` is empty
   throughout. That test is from
   [[experiments.database.pputida-cell-state]] (worktree
   `docs/pputida-cell-state`), not mine. Corroborating it from the loader side:
   `cycle` appears there only as a build-time dict key
   (`by_strain[(construct, cycle)]`) and inside verification diagnostic labels
   (`label=f"{construct} DBTL{cycle}"`, `carruthers2025.py:5019,6217`), never as
   a field on a stored record. **This blocks MVP criterion 2 outright**, since
   cycle extrapolation needs the cycle. It is also what blocks the vector
   token, because the vector is a function of the cycle. Fixing this is the
   first task of the next session.
2. **No STRING for any bacterium.** `torchcell/graph/graph.py` is yeast-only and
   hard-codes the `4932.` prefix. Taxon 160488 is already known to the codebase
   from the GOA spec but never used for STRING.
3. **No bacterial regulatory graph.** No RegulonDB, Abasy, CollecTF or
   RegPrecise ingestion exists. The only bacterial TRN in the repo is an
   unopened RegulonDB file inside the Lamoureux 2023 release, classified in
   `notes/plan.bacteria-si-phenotype-audit-ecoli.md` as annotation, not a
   measurement.
4. **iJN1463 is not in the repo or any data tier.** Every metabolic-module code
   path hard-codes `YeastGEM` / yeast-GEM 9.0.2.
5. **ESM2 and ProtT5 are typed to `SCerevisiaeGenome`.** Only GenSLM, codon
   frequency, one-hot and random are genome-agnostic today, and
   `NodeEmbeddingBuilder.EMBEDDING_CONFIGS` has no `data/pputida/...` roots.
6. **No HiGenLM, HiSan, Evo or Evo 2 code anywhere.** One aspirational mention
   in `notes/paper.north-star.md:74`.

### The campaign: 7 DBTL cycles, and cycle 3 is cycle 2 on a new vector

Every number in this section and the two that follow is regenerated by
`experiments/048-pputida-isoprenol-cgt/scripts/carruthers_campaign_audit.py`,
which writes `results/carruthers_campaign_audit.json`,
`results/design_reuse_across_cycles.csv`, `results/vector_per_cycle.csv` and
`results/titer_by_cycle.csv`. Measured from `Carruthers_etal_2025.csv`:

| quantity | value |
|---|---|
| rows (line x replicate) | 1,506 |
| DBTL cycles | 0 through 6 |
| distinct designs (target sets) | 402 |
| distinct gene targets | 121 |
| replicates per design | 3 for 458 of 472 design-cycle cells |
| no-gRNA controls | 12 per cycle, 18 in cycle 0 |
| sgRNAs per design | 1 in cycle 0, then 2 to 4 |

Design reuse across cycles is negligible everywhere except one pair. Cycle 2
and cycle 3 share **50 designs**; no other cycle pair shares more than 7. That
is the plasmid-swap experiment, and the released plasmid table names the swap:

| cycles | vector | description (SI Table 3) |
|---|---|---|
| 1, 2 | `pIY989` (JBx_249165) | `pRSF1010-Gm-NagR-pNagAa-dCas9-thrLABC-sfGFP` |
| 3, 4, 5, 6 | `pDBTL3-6_Vector` (JBx_273384) | `pRSF1010-Gm-L7-NagR-pNagAa-dCas9-thrLABC-sfGFP` |

The two descriptions differ by the `L7` linker ahead of `NagR`. On the 50 shared
designs, measured on mean titer per design:

| statistic | value |
|---|---|
| cycle 2 vs cycle 3, Pearson | 0.670 (n = 50) |
| cycle 2 vs cycle 3, Spearman | 0.668 |
| same, on fold-over-control | 0.670 |
| mean paired difference | -2.5 mg/L (312.9 vs 310.3) |
| within-cycle split-half reliability, cycle 2 | 0.946 [0.926, 0.973] |
| within-cycle split-half reliability, cycle 3 | 0.980 [0.974, 0.985] |
| mean dCas9 signal, cycle 2 vs cycle 3 | 2.63e6 vs 4.31e6 (1.6x) |

The reliability numbers are Spearman-Brown corrected split-half over 400 random
replicate splits. They are the decisive part: the replicate ceiling is 0.95 to
0.98, and the same designs on the new vector agree at only 0.67. **The vector
change is signal, not noise.** It leaves the mean titer essentially unchanged
and reorders the strains, which is exactly the behavior a context variable has
and exactly what a design-only model cannot represent. A model trained on the
`pIY989` cycles and tested on the `pDBTL3-6_Vector` cycles is capped near 0.67
no matter how good its genotype representation is.

That makes the vector a first-class input, not a nuisance. The cheapest honest
encoding is the 030 ontology-token route: a learned vector token added to the
design representation, two levels here, which is the binary mapping already
anticipated. The more faithful encoding is the sequence of the vector itself,
which we hold (see the provenance table below), and which generalizes to a
third vector where a binary token does not.

### Provenance audit of the shared folder

Juan's folder is `/tmp/screenshots/04_Data`. Its phenotype content is fully
reproducible from the already-mirrored published record. Verified: all 1,506
titer rows in `Carruthers_etal_2025.csv` match the published SI
(`si9.xlsx`, sheets `Figure 1D` and `Figure 4b`) with **maximum absolute
difference 0** on 3,012 matched comparisons, 0 unmatched. So the CSV is
a derived analysis-ready join, not a new measurement, and the loader must source
from the mirror, not from the CSV.

Mirror key: `carruthersAutomationMachineLearning2025`, already present in both
`$DATA_ROOT/torchcell-library/` (paper, OCR, si1 through si9) and
`$DATA_ROOT/torchcell-raw/` (SI xlsx plus the Dryad deposit). The raw-mirror
provenance is already complete for the published part: the Dryad deposit carries
DOI `10.5061/dryad.gtht76hzh`, version `v20250820`, retrieved by hand on
2026-10-09 under `RetrievalMethod.manual_browser` because Dryad serves an Anubis
JS proof-of-work to scripts. Per-cycle raw DIA sits in PRIDE
(`PXD063733`, `PXD063737`, `PXD063738`, `PXD063740`, `PXD063743`, `PXD063744`,
`PXD063746`), recorded in the loader as `PRIDE_ACCESSIONS` and read by nothing.

Already in the mirror, and the existing loader already reads it:

| content | mirror path |
|---|---|
| per-line titers, cycle, control flag, QC filter | `torchcell-library/.../si/si9.xlsx` |
| 121 gene targets, source, RbTnSeq presence, dCas9/POI ratios, cycle usage | `.../si/si4.xlsx` (SI Table 1) |
| all plasmids with JBEI accessions and descriptions | `.../si/si6.xlsx` (SI Table 3) |
| the 361 CRISPRi arrays by cycle, construct, sgRNA count, vector | `.../si/si7.xlsx` (SI Table 4) |
| oligonucleotides | `.../si/si5.xlsx` (SI Table 2) |
| proteome, per line per replicate, ~1,700 UniProt columns, with titer | `torchcell-raw/.../data/dryad/CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data.csv` (1,496 rows) |
| full environment record | `torchcell-raw/.../data/dryad/CRISPRi_automation_Pputida_proteomic_metadata.csv` (1,497 rows) |

The environment record is complete enough to fill `Environment.media` without a
gap: M9-NREL, glucose 20 g/L, 24 C, 1000 rpm, 48-well BioLector flower plate,
L-arabinose 2 g/L induced at 8 h, assayed at 48 h.

**Not in the mirror, and therefore owed to the data lake.** These are the files
Juan shared that the publication does not release as sequence, so each needs a
raw-mirror entry with `source_url` (or a manual recipe naming Juan and the date
received), `retrieval_method`, `sha256` and `retrieved_at`:

| file | what it is | note |
|---|---|---|
| `dbtl2-1.gb` | 11,046 bp dCas9 + sgRNA array plasmid, cycle 2 example | RSF1010 oriV, `aacC1` Gm, `pnagAa`-dCas9, J23119 array, BASIC scars |
| `dbtl3-1.gb` | 11,046 bp, cycle 3 example | differs from `dbtl2-1.gb` at only 29 nt, all in the spacer region from 1,347 |
| `dbtl0-2_plasmid.fasta` | 9,959 bp "non-sgRNA plasmid used in DBTL0-2" | |
| `dbtl3-6_plasmid.fasta` | 14,066 bp, header says DBTL0-2 but the filename says DBTL3-6 | header is inconsistent, resolve with Juan before pinning |
| `86kb_deletion.csv` | 58 genes in the host's 86 kb deletion | corroborates the `GenomicSpan` the loader already carries (86,812 bp, 4,538,575..4,625,386), so this is a cross-check, not a new source |
| `pputida_isoprenol.json` | iJN1463 plus the isoprenol route | diff below; check whether the paper's GitHub releases it before treating it as unpublished |

Two things to resolve with Juan rather than guess. First, the
`dbtl3-6_plasmid.fasta` header names DBTL0-2 while its length (14,066) differs
from the DBTL0-2 file (9,959) by 4,107 bp, so one of the two labels is wrong.
Second, SI Table 3 gives a JBEI accession for every plasmid, so part of this may
be retrievable from the JBEI registry, which would be a better provenance route
than a hand-received file.

### The metabolic model, and why the readout is native to it

`pputida_isoprenol.json` is iJN1463 with the isoprenol route added. Measured
diff, from the same audit script:

| | iJN1463 | isoprenol model |
|---|---|---|
| reactions | 2,927 | 2,958 |
| metabolites | 2,153 | 2,178 |
| genes | 1,462 | 1,474 |
| objective | `BIOMASS_KT2440_WT3` | `BIOMASS_KT2440_WT3` |

The added genes are `mvaE`, `mvaS`, `MK`, `PMDHKQ`, `aphA` (heterologous, no
`PP_` id) plus 8 native `PP_` genes. The added reactions carry the mevalonate
route to isoprenol (`HMGCOAS`, `HMGCOAR`, `MEVKIN`, `PMD`) and, importantly,
`EX_ip_e` with `ip_c`, `ip_p`, `ip_e`. So the phenotype we predict is a native
exchange reaction of the model, which is the condition 031 wanted and did not
get for betaxanthin: the metabolite readout is a node in the metabolic module,
not an external regression target bolted on.

The heterologous genes are also the pathway plasmid's content, which closes the
loop with the sequence-level genotype principle: pathway plasmid, dCas9 vector,
and host 86 kb deletion are all edits to the total genomic content in the cell,
and all three are things we hold sequence for.

### Dataset size, and why this is a transfer problem

The supervised target is small: 465 served titer records over 402 designs and
121 targets. On its own that will not train a CGT, and a transformer losing to a
boosted tree on 402 designs is the expected outcome, not a surprise, which 030
already measured in yeast.

What makes the program viable is that the P. putida side of the store is not
small. The two largest served P. putida datasets are
`RbTnseqBorchert2024Dataset` at **1,466,920 records** and
`FitnessBrowserPutidaDataset` at **1,118,052**, plus the 321-sample transcriptome
compendium `PutidaPrecise321Lim2022Dataset`. So this follows the 040 pattern,
general model then fine-tune: pretrain the gene representation on the millions of
P. putida fitness records, then fit the isoprenol head on 465 records. The
metabolic model, the STRING graph and the sequence embeddings carry the rest of
the inductive bias.

Stated plainly as a hypothesis, because none of it is measured yet: **I expect
the isoprenol head to need the fitness pretraining to beat the lower-order null,
and I have not tested this.** The MVP is designed to find out, which is why the
null is on the same table as the model.

### Sequence windows for a bacterium: the ORF, not a 1 kb promoter window

The recollection of "1,000 upstream, 300 downstream" is the yeast setting
(`nt_window_five_prime_1003`, `nt_window_three_prime_300` in
`torchcell/datasets/nucleotide_transformer.py`). Measured on the pinned KT2440
GFF by `scripts/kt2440_sequence_window_budget.py`, it does not transfer:

| statistic over 5,729 KT2440 genes | value |
|---|---|
| median clear upstream gap to the neighboring gene | 88 bp |
| mean clear upstream gap | 130 bp |
| genes overlapping or abutting the upstream neighbor | 17.4% |
| genes with 100 bp clear upstream | 46.9% |
| genes with 300 bp clear upstream | 9.8% |
| genes with 1,000 bp clear upstream | **0.7%** |
| gene length, median / q99 / max | 834 / 3,431 / 26,049 bp |

For 99.3% of KT2440 genes a 1 kb upstream window is mostly the neighboring
gene's coding sequence, which is what an operonic genome implies. So a fixed
promoter window is the wrong sequence unit here, and the right one is the ORF
itself, with at most a short (100 to 200 bp) upstream flank that is understood
to reach into the neighbor.

That is also why FUDT genuinely does not apply, as the owner noted, and why the
protein-level and codon-level encoders carry more weight for this organism than
a DNA promoter-window model. GenSLM already does the right thing without
modification: its window is `("cds", GENSLM_SEQ_LENGTH * 3)` = 6,144 nt
(2,048 codons), which covers well past the 99th percentile gene, and its
`__main__` already runs P. putida KT2440.

### How to capture the promoter: at the transcription unit, not the gene

The constructive half of the window question. In a bacterium the promoter
belongs to the transcription unit, so the capture is: chain adjacent same-strand
genes into TUs, take a promoter window upstream of the TU **leader** only, and
let every gene in the TU inherit its leader's promoter.

The operon threshold is measured, not assumed, by
`scripts/kt2440_transcription_unit_promoters.py`. The negative class is
**divergent** gene pairs, which cannot share a TU and must hold promoters in
their gap; convergent pairs are reported as the terminator-containing control.

| median intergenic gap, by orientation | value |
|---|---|
| tandem (same strand, n = 4,014) | 42 bp |
| convergent (n = 857) | 72 bp |
| divergent (n = 857) | **178 bp** |

A gap that has to hold two promoters is 4.2x a tandem gap. The threshold scan
shows how sharply that separates:

| tandem gap at or below | tandem | divergent | enrichment |
|---|---|---|---|
| 0 bp | 0.240 | 0.018 | 13.7x |
| 20 bp | 0.423 | 0.028 | **15.1x** |
| 40 bp | 0.496 | 0.041 | 12.1x |
| 100 bp | 0.689 | 0.173 | 4.0x |
| 150 bp | 0.793 | 0.400 | 2.0x |
| 300 bp | 0.937 | 0.823 | 1.1x |

So a tandem gap under about 40 bp essentially never holds a promoter, and past
roughly 150 bp the distinction is gone. Taking 40 bp as the chaining threshold:

| result | value |
|---|---|
| transcription units from 5,729 genes | 3,740 |
| multigene units | 1,107 (largest 14 genes) |
| promoter window upstream of a TU leader, median available | **151 bp** |
| units with no promoter window at all | 3 |
| units hitting the 300 bp cap | 581 |

That is the payoff: per gene, the median clear upstream is 88 bp and about a
thousand genes have none at all. Per TU leader it is 151 bp and **three** units
have none. Moving to the TU nearly doubles the promoter sequence available and
removes the zero-window problem almost entirely.

Rules that follow, and they are implementation requirements, not preferences:

- The window is **clipped, variable-length, and never padded with coding
  sequence**. The available length is itself a feature, so the model can tell a
  200 bp promoter from a leader that abuts its neighbor instead of being handed
  a silently contaminated window.
- Cap at 300 bp, near the divergent median, since that is roughly how much
  sequence a gap that genuinely holds a promoter offers.
- Genes in one TU **share** a promoter representation. This also matters
  mechanistically for CRISPRi: a guide against an internal gene of a 14-gene
  operon is not the same perturbation as one against the leader, and the TU
  assignment is what lets the model express that.
- The threshold is a parameter with a measured basis, not a constant to be
  tuned quietly. 20 bp maximizes separation, 40 bp trades a little separation
  for longer units; report which was used.

A caveat on all of the above: this is **predicted** operon structure from gene
adjacency, not measured transcription units. The honest upgrade is a TSS or
operon measurement. The already-served
`PutidaPrecise321Lim2022Dataset` (321-sample transcriptome compendium) can test
the TU calls directly by within-TU versus across-TU co-expression, which is a
real validation available without new data. Do that before leaning on the
predicted TUs for anything load-bearing.

### Promoters in this dataset vary on the plasmid, not the chromosome

A distinction that matters for how promoter sequence is used here. CRISPRi does
not edit promoters, so **native promoter sequence is constant across every
strain in 048**. It is a per-gene node feature, useful through pretraining and
transfer across the millions of P. putida fitness records, and it cannot by
itself explain any variation in isoprenol titer.

What varies per record is the engineered expression context: the sgRNA array
promoter (`J23119`, constant), the dCas9 promoter (`pnagAa`, salicylate
inducible, constant), the pathway plasmid promoters (`PBAD`, `Ptrc1-O` on
pIY670), and the one that actually changes, the **vector**, where SI Table 2
lists `Linker7_Insert_F/R` as a 70 nt insert, corroborating that the
`pIY989` to `pDBTL3-6_Vector` difference is the L7 linker ahead of `NagR`.

**A gap to raise with Juan:** the sgRNA spacer sequences are not in the released
tables. SI Table 2 holds check primers, SI Table 4 names the arrays by target
set, and the only spacers we hold are in the two example genbanks, where
`dbtl2-1.gb` and `dbtl3-1.gb` differ at 29 nt in the spacer region. Without the
spacer table we cannot compute guide position relative to the TU leader or the
TSS, which is likely the strongest promoter-adjacent feature available for a
CRISPRi dataset, since repression strength depends on where dCas9 binds. Ask
for the spacer table or the full plasmid set.

Planned encoder set, grouped by what it reads and in the order the sanity
checks come first:

| encoder | reads | status in repo |
|---|---|---|
| one-hot | ORF | genome-agnostic, works today |
| random (several seeds) | nothing, control | genome-agnostic, works today |
| codon frequency | CDS | genome-agnostic, works today |
| **GenSLM 25M / 250M / 2.5B PATRIC** | CDS, 6,144 nt | genome-agnostic, bacterial-trained, already runs on KT2440: **the owner's pick to start, and the fast path** |
| **HiGenLM / HiSan** | nucleotide, long context | the owner's other pick; no code, needs a new dataset class and weights from ANL |
| ESM2 | protein | typed to `SCerevisiaeGenome`, needs the type relaxed |
| ProtT5 | protein | typed to `SCerevisiaeGenome`, needs the type relaxed |
| CaLM | CDS codons | `CalmDataset` exists, CDS-based |
| Nucleotide Transformer | ORF rather than flanks | class exists, windows are yeast-tuned |
| Evo or Evo 2 | nucleotide, long context | no code, new dataset class |

One-hot plus the randoms are the floor the whole table is read against, and they
run first, before any new model is added, so the pipeline is proven end to end
on encoders that cannot be blamed.

### Core reusable objects versus experiment-local assembly

The division the owner asked for: reusable objects in core, the dataset that
combines them in the experiment. Following the 031 precedent, where
`torchcell/models/cell_graph_transformer_metabolism.py` subclasses the CGT
unchanged and the whole study lives in `experiments/031-betaxanthin-module/`.

**Goes in core (`torchcell/`), because a second organism will need it:**

| new object | where | why core |
|---|---|---|
| a host-parameterized STRING builder | `torchcell/graph/` | `graph.py` hard-codes `4932.`; the taxon, the id-to-locus-tag mapping and the channel set are the only organism-specific parts |
| `PPutidaGraph` (or a generalized `<Host>Graph`) | `torchcell/graph/` | `torchcell/graph/` currently has no bacterial builder at all |
| `BiggGEM` beside `YeastGEM` | `torchcell/metabolism/` | must expose the same `model` / `gene_set` / `reaction_map` / `bipartite_graph` surface so `neo4j_cell.py` takes it unchanged |
| genome-agnostic `Esm2Dataset` / `ProtT5Dataset` | `torchcell/datasets/` | both are typed to `SCerevisiaeGenome` today; the fix is a type relaxation, not a new class |
| new embedding datasets (HiGenLM, Evo, NT) | `torchcell/datasets/` | each implements `MODEL_TO_WINDOW`, `initialize_model`, `process` per `BaseEmbeddingDataset` |
| `data/pputida/kt2440/...` embedding roots | `node_embedding_builder.py` | `EMBEDDING_CONFIGS` is all `data/scerevisiae/...` |

**Stays in `experiments/048-pputida-isoprenol-cgt/`:** the query, the dataset
assembly that joins titer + proteome + graphs + embeddings, the split
construction (design-level random and cycle extrapolation), the vector-token
arm, the baselines, the train script, the conf yamls, the readers and the
figures.

### MVP end criterion

The overnight goal, and the contract this note exists to hold stable:

1. A trained model on the 048 dataset under **random split 0.8 / 0.1 / 0.1 over
   designs** (split by design, never by replicate row, so replicates of one
   design never straddle the split).
2. The same model under **cycle extrapolation: train on cycles 0 through 5,
   predict cycle 6**. Only the last cycle is predicted from the previous ones.
   Earlier-cycle-ahead variants are out of scope for the MVP.
3. Both readouts reported as validation and test Pearson and MSE on isoprenol
   titer, train and validation curves on one axis, logged to W&B, with the
   lower-order null (per-target mean, additive, boosted trees) on the same
   table.
4. The vector arm reported both ways: with and without the vector token, since
   the 0.670 against a 0.95 ceiling says the ablation is the most informative
   single comparison in the whole MVP.

Done means those four, with every artifact produced by a committed script in
`experiments/048-pputida-isoprenol-cgt/scripts/`.

Not in the MVP: the proteome co-head (it follows, as 046 does for yeast), the
kinetic layer, inverse design, and any new wet-lab proposal.

**Criterion 2 is blocked until the DBTL cycle reaches the served graph.** That
is gap 1 above and it is the first task, not a parallel one: without the cycle
there is no cycle split and no vector token, so criteria 2 and 4 cannot be
evaluated at all. Criteria 1 and 3 can proceed on the served titer records
meanwhile.

### The proteome pairing is better here than in yeast

Worth recording because it shapes what comes after the MVP.
`CampaignProteomeCarruthers2025Dataset` serves 465 records and
`IsoprenolTiterCarruthers2025Dataset` serves 465 campaign records, measured on
the same cultures: the Dryad quantification table carries `Isoprenol_titer` in
the same row as the ~1,700 protein columns. So proteome and phenotype are
paired one-to-one, which is stronger than the yeast situation where the omics
co-targets (Kemmeren, Messner) are gene-keyed and only partially overlap the
phenotype pool.

That makes the proteome co-head the natural first extension after the MVP, and
it can be a true co-head on the same record rather than a separately keyed
auxiliary target. It stays out of the MVP so that criterion 1 is not confounded,
but it is the obvious next arm.

### Logging and reporting conventions

Carried over so later sessions do not relitigate them:

- Train and validation loss on one axis, always.
- Per-label loss and performance logged, not just the aggregate.
- W&B views over reports, with a Charts view each round; group links per arm.
- Every number that reaches this note or a figure comes from a committed script
  in `experiments/048-pputida-isoprenol-cgt/scripts/`, reading real result
  files.
- Figures follow the repo standard (`PLOT_PALETTE`, Arial 6 pt, boxed axes,
  panel widths from `PANEL_WIDTHS_MM`, true-size SVG embedded in the note).

### Ordered plan across sessions

1. Materialize the DBTL cycle (and therefore the vector) on the Carruthers
   records. Decide with the owner whether this is a loader change plus a
   superset admission, or an experiment-local join on construct name against the
   raw mirror's `DBTL_Cycle`. The loader route is the correct one by the
   constitution; the join is the fast one. **Needs the owner's call, since a
   loader change to a served dataset triggers the admission check.**
2. Land `BiggGEM` with iJN1463 into the data tier, pinned by sha256 like the
   genomes tier, and decide whether the isoprenol-augmented model is a second
   pinned artifact or a documented derivation from it.
3. Bacterial STRING: taxon 160488, the channel set, and the STRING-protein to
   `PP_` mapping.
4. Embeddings. Owner's call 2026-11-17: **start with GenSLM, then HiSan.**
   GenSLM needs no new code (its `__main__` already runs KT2440), so it goes
   first alongside one-hot and the randoms as the floor, and the pipeline is
   proven end to end before HiGenLM/HiSan is wired as a new
   `BaseEmbeddingDataset` subclass.
   Also in this step: the TU promoter windows from
   `kt2440_transcription_unit_promoters.py` as a per-gene feature, and the TU
   co-expression validation against `PutidaPrecise321Lim2022Dataset`.
5. The dataset assembly, the splits, the baselines, then the model.

Each step ends with its numbers in this note under a new dated H2, not in a
conversation.
