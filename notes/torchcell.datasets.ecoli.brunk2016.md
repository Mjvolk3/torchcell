---
id: 31tf2rwzxrrl7mgovvcxu2e
title: Brunk2016
desc: ''
updated: 1791616841335
created: 1791616841336
---

## 2026.10.10 - Four arms of one fermentation, and what each one serves

Brunk et al. 2016 (Cell Systems, doi:10.1016/j.cels.2016.04.004) is row 58 of the
bacterial candidate table. It sampled ONE batch fermentation of nine strains over 0 to
72 hours: "three isopentenol-producing strains (I1-I3), three limonene-producing
strains (L1-L3), two bisabolene-producing strains (B1-B2), and wild-type E. coli DH1
(WT)". Four quantities are released per sample, on four scales, so four dataset classes.

| dataset | class | records | keys | measurement_type |
|---|---|---|---|---|
| `metabolome_brunk2016` | `MetabolomeBrunk2016Dataset` | 117 | 51 COBRA ids | `lc_ms_concentration_um` |
| `exometabolite_brunk2016` | `ExometaboliteBrunk2016Dataset` | 126 | 6 COBRA ids | `hplc_extracellular_concentration_g_per_l` |
| `proteome_brunk2016` | `ProteomeBrunk2016Dataset` | 81 | 44 MG1655 loci | `srm_protein_peak_area` |
| `biofuel_titer_brunk2016` | `BiofuelTiterBrunk2016Dataset` | 72 | 3 products | g/L titer |

No schema change: every class already existed, and `Environment.duration_hours` is what
makes 14 samples of one strain 14 records instead of one.

### Retrieval

Scriptable on both routes, measured 2026.10.10 by
`experiments/036-dataset-fixes-before-kg-build/scripts/brunk2016_release_inventory.py --network`:
`mmc1.pdf`, `mmc2.xlsx` and `mmc3.xlsx` come off the Elsevier CDN (HTTP 200, no
challenge) and the article's full text off the PMC author-manuscript bucket
(`PMC4882250.1/PMC4882250.1.txt`, HTTP 200). All five artifacts, the MinerU OCR of the
supplementary PDF included, are deposited in
`$DATA_ROOT/torchcell-raw/brunkCharacterizingStrainVariation2016/` with sha256 and
retrieval records. Nothing went into Zotero. `is_pmc_openaccess` is false for this id,
so there is no PDF in the bucket and the quote anchor is the publisher's own plain text.

### Table S1 is a raster image, and the OCR shifts its last cells

The strain-to-plasmid table exists only as an image inside `mmc1.pdf`; `pdftotext`
returns nothing for it. MinerU read it at 200 DPI and again at 350 DPI: the two passes
agree byte for byte on the strain block, and BOTH put `JPUB_002460 + JPUB_002466` on the
`DH1` row while leaving `B2` empty, a one-row cell shift.

The repair uses no image and no eye. The article calls DH1 "wild-type E. coli DH1 (WT)",
a wild type carries no production plasmid, every listed plasmid must be on some strain,
and exactly one strain row is empty, so the orphaned pair is B2's. `read_table_s1`
refuses any other shape. The proteome arm's L4 re-asserts the repair against a different
file: every protein that plasmid pair encodes and the workbook measures is higher in B2
than in DH1, the smallest ratio being 7.4x (PMK) and the largest 75x (bisabolene
synthase). The weaker claim "DH1 is the lowest strain for every pathway protein" is NOT
made, because the bytes refute it: DH1's GPPS area (26,219) is above I1's (2,896), which
is what a background signal for a gene a strain does not carry looks like.

### Genotypes are parsed from the plasmid names, and the organisms are read, not guessed

A plasmid's released name IS its part list (`pBbA5c-MevTsa-MK-PMK`), so a strain's
genotype is the gene tokens of its one or two plasmids, each a
`HeterologousPathwayPerturbation`. `MevT` is expanded because the SI expands it, "the
modulation of protein expression in the 'top' portion of the mevalonate pathway (i.e.,
atoB, HMGS, and HMGR, aka 'MevT')", with the three forms typed from the same paragraph.
`source_organism` comes from the proteomics workbook's own `Organism` column, which
`check_pathway_organisms` re-reads at build time (11 genes settled): E. coli for AtoB,
Idi, IspA and NudB, S. cerevisiae for HMGS, HMGR, MK, PMK and PMD, S. aureus for the
`sa` pair. GPPS, LS and BIS have no organism anywhere in the mirror and take the
unreported sentinel. The four E. coli genes are extra copies of NATIVE genes and are
stored under their MG1655 b-numbers; DH1's own lesions are never written by the paper,
so the background carries no alleles.

### Replication, and why no dispersion is stored

The served sheets release ONE number per strain-hour. The two triplicate sheets cover
different grids (endometabolomics: 4 strains x 5 hours inside the first 3 h; proteomics:
4 strains x 6 hours), so they are not those numbers' dispersion, and the SI says plainly
what the paper did where it had none: "For metabolites or peptides that did not have a
triplicate measurement, we estimated the variance using the average variance for all
metabolites or peptides measured". That is a derived quantity. Every record therefore
carries `n_replicates = 1` per key and a typed gap on its SE map.

### Drops, each counted

| rule | items |
|---|---|
| column header states no unit | 26 (20 amino acids, Cystine, 5 DXP intermediates) |
| protein key is not resolvable to one host locus | 24 of 68 host SRM proteins |
| protein is not a host protein | 13 (pathway enzymes, AmpR, Cam, BSA) |
| sample hour label is not a stated time | 8 (`72C`) |
| sample measures no column of this unit block | 7 metabolome samples (hour 2) |
| sample shares no metabolite with the wild type of its hour | 2 (I1 and I3 at hour 2) |

The protein refusal is the one worth watching: the release keys a protein by an internal
JBEI id, and only two released mappings key such an id to ONE gene, the UniProt
accession plus `GN=` in the triplicate sheet (33 proteins) and a single-gene GPR in the
identifier sheet (20), 44 distinct between them. The other 24 carry only a multi-gene
GPR (`FRD2` is `b4151 and b4152 and b4153 and b4154`), which does not say which subunit
the measured peptide belongs to. The per-protein ledger is in
`preprocess/protein_keys.csv`, and the gap is filed as a dataset issue rather than
closed with a guessed subunit.

### Duplication against the landed E. coli proteomes: independent

Measured by the inventory script. `proteome_brunk2016` shares NO record key with
`proteome_schmidt2016` or `proteome_ishii2007`: those are pinned to BW25113 and keyed by
its locus tags, this one to MG1655 and keyed by b-numbers, so the locus overlap is 0 in
both cases. Joined instead on each assembly's own gene symbol, the wild-type profiles
correlate r = 0.6153 over 44 shared symbols with Schmidt 2016 (absolute copies per cell)
and r = 0.7729 over 32 with Ishii 2007 (mg protein per g dry cell weight). Different
host strain, different assembly pin, different measurement type and different cultures,
with a consistent abundance ordering: three independent measurements, not a re-serving.

### Verification

| dataset | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| `metabolome_brunk2016` | 117 records validated | 117 = 117; 117 unique (strain, environment) pairs | 3,820 values finite | measurement_type single; hour in window; reference is the same-hour wild type | 3,820 values re-read from the workbook agree exactly |
| `exometabolite_brunk2016` | 126 records validated | 126 = 126; 126 unique pairs | 756 values finite | same three rows | 756 values agree exactly |
| `proteome_brunk2016` | 81 records validated | 81 = 81; 81 strain-hour samples | 3,564 values finite | same three rows | 81 sample totals agree; the OCR repair holds against the proteomics workbook |
| `biofuel_titer_brunk2016` | 72 records validated | 72 = 72 | 72 titers >= 0 | g/L verbatim; every uncertainty a typed gap; isopentenol is the canonical isoprenol entity | 72 titers agree exactly |

The metabolite gate's strain signature now also carries each perturbation's `variant`
and `construct_name`: I1 and I2 carry the same seven genes and differ only in whether
the HMGS and HMGR copies are the original or the codon-optimized ones, so without those
fields they collide in L1. Adding fields can only separate records that were colliding,
so no dataset that passes that rule today can start failing it.

Schema impact: `scripts/schema_impact_check.py --base origin/main` reports no schema
contract changes, so this is additive and no served dataset is touched.

## 2026.10.10 - Keying the multi-GPR SRM proteins by their released peptides (#872)

The 24 host proteins refused above carry a JBEI id and a multi-gene GPR only, but the
proteomics sheet (`data/mmc3.xlsx`, sha256 `c8161e23...41ae`, sheet `Raw proteomics
data`, column `Peptide`) releases the measured peptide of every row. A third route now
searches those peptides against the MG1655 protein FASTA the genomes tier deposits
(`GCA_000005845.2_ASM584v2_protein.faa.gz`, sha256 `900cb656...5263`, 4,290 proteins
keyed to b-numbers by `read_protein_fasta`, resolved through `registry.resolve`).

The rule (`resolve_by_peptides`): a protein is keyed to locus `L` only when EVERY one of
its released peptides occurs in `L` and in no other MG1655 protein. Leucine and
isoleucine are read as one residue and no cleavage rule is assumed, because an SRM
transition cannot separate isobaric sequences; both choices can only add matches, so the
uniqueness call is conservative. A peptide that occurs in two proteins refuses the whole
protein, because the released `ProteinArea` is the mean of the protein's corrected
peptide areas (PFLB, DH1 at 24 h: peptides 659,242 and 699,128, `ProteinArea` 679,185),
so one shared peptide makes the area not one gene's.

### Result, measured from the rebuilt dev store

| | before (PR #873) | after |
|---|---|---|
| records | 81 | 81 |
| distinct protein keys | 44 | 65 |
| stored peak-area values | 3,564 | 5,265 |

Routes over the 68 host proteins: 33 UniProt `GN=`, 11 single-gene GPR (20 proteins
have one, 9 of them already keyed by UniProt), 21 by peptide, 3 refused.

Keyed by peptide (21): ACKA b2296, ADHE b1241, AtoB b2224, DHSC b0721, DHSD b0722, EUTD
b2458, FDHF b4079, FRDA b4154, FRDB b4153, FRDC b4152, FRDD b4151, GLPX2 b2930, HYCB
b2724, HYCC b2723, HYCD b2722, HYCE b2721, HYCF b2720, LACI b0345, NudB b1865, ODO2
b0727, PTA b2297.

Refused, each with `REFUSAL_SHARED_PEPTIDE` (3):

| protein | peptide matches (I = L) |
|---|---|
| DHSB | `FLIDSR`: b0724 and b3876 (as `FLLDSR`, a non-tryptic position); `LDGLSDAFSVFR`: b0724 |
| HYCG | `HADILLFTGAVTR`: b2719 (hycG) and b2489 (hyfI), identical |
| PFLB | `VDDLAVDLVER`: b0903; `YPQLTIR`: b0903 (pflB) and b2579 (tdcE), identical |

DHSB is the one owner call: it is refused only because of the I = L equivalence at a
position trypsin would not cut in b3876. Requiring a tryptic context would key it to
b0724; that is not done, because it assumes complete enzyme specificity.

### Sequence cross-check of the name routes

`check_peptides_agree` now asserts, at build time, that every keyed protein's locus
carries every one of its released peptides: 65 of 65 do. Two proteins keyed by UniProt
pass that check but have peptides that also occur in a paralog, so their released area
may carry the paralog's signal: FUMA (both peptides in fumA b1612 and fumB b4122) and
TKT1 (`ALSMDAVQK` in tktA b2935 and tktB b2465). They keep the released UniProt key; this
is recorded, not acted on.

### Verification (rebuilt with `--retire-existing`, `verify_build(..., family="proteome")`)

| dataset | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| `proteome_brunk2016` | 81 records validated | 81 = 81; 81 strain-hour samples | 5,265 values finite | single measurement_type; hour in window; reference is the same-hour wild type over a subset of its keys | 81 sample totals agree within 1e-6 against the workbook; the OCR repair holds (9 of 9 proteins higher in B2, smallest ratio 7.4x) |

L4 now reads WHICH proteins are keyed from the build's own `preprocess/protein_keys.csv`
(the peptide route needs the assembly) and re-reads every area from the pinned workbook.
`build_dataset_lmdb --list-stale --include-private` no longer names any Brunk 2016 store.

Schema impact: `scripts/schema_impact_check.py --base origin/main` reports "No schema
contract changes vs origin/main". The record classes are unchanged; the store's key set
grows, so the served graph takes it in the KG 4.0 full rebuild.
