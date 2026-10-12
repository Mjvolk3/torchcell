---
id: n5d2o2d4lksyw71iotqugz3
title: Zhang2020
desc: ''
updated: 1791808976165
created: 1791808976165
---

## 2026.10.12 - Zhang 2020: the schema is landed, the loader is BLOCKED on the genome tier

Issue #990 named three gaps. Schema PR #1055 closes the two that are schema gaps; the
third is outside the schema and still blocks the loader, so NO Zhang 2020 loader is in
that PR.

### Closed by the schema

1. **Which promoter drives a gene.** `PromoterDrivenGeneIntegrationPerturbation`, a new
   leaf under `GeneAdditionPerturbation`: the driven gene is the perturbation's identity,
   `promoter_name` the part driving it, `integration_locus` REQUIRED (the parent leaves it
   optional, because a plasmid-borne addition has none), `promoter_sequence` optional for a
   named part the release does not deposit. Zhang 2020 varies exactly this: "five genes
   selected from GSM simulations and a priori biological understanding, each controlled by
   six different promoters from a set of 30 promoters selected from transcriptomics data
   mining", integrated at EasyClone sites ("All integrations were directed toward EasyClone
   sites").
2. **The biosensor-proxy readout.** `BiosensorFluorescencePhenotype` + its experiment pair:
   the released number is a specific GFP synthesis rate in MFI/h from a sensor that reports
   intracellular tryptophan ("a yeast tryptophan biosensor was developed based on the trpR
   repressor of the trp operon from Escherichia coli"), so the leaf keeps the signal in its
   own units and names the sensed metabolite as a typed `Compound`. It is not
   `PromoterActivityPhenotype` (that would attribute the signal to the varied promoters,
   which no one measured), not `EnvironmentResponsePhenotype` (whose verifier requires an
   environmental edit on every record), and not `MetabolitePhenotype` (which holds a
   metabolite QUANTITY, so a fluorescence rate stored there would pose as a titer).

The third gap of the issue, `ProductTiterExperimentReference.genome_reference` being an
`AssemblyReferenceGenome` with a bacterial-only `assembly_set`, is untouched: the 6 HPLC
titers are blocked behind the same genome-tier question as the rest of the dataset, and a
yeast product-titer pair is its own decision.

### The block, measured

The candidate gate still reads **blocked:provision** at G2
(`database/candidates/zhangCombiningMechanisticMachine2020.json`): `gates.HOSTS` maps the
`CEN.PK` token to no assembly set, and the paper's host is CEN.PK2-1C ("S. cerevisiae
strains were derived from CEN.PK2-1C"). What was measured today, by
`experiments/database/scripts/vaishnav2022_release_inventory.py --cenpk`:

- the candidate assembly is **GCA_002571405.2 (ASM257140v2)**, from
  `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/571/405/GCA_002571405.2_ASM257140v2/`
- its GenBank flat file's md5 `bb54576a3b512b8499af9af555952918` MATCHES the directory's
  own `md5checksums.txt`, so the retrieval is scriptable with a recorded `direct_url`
  (sha256 of the retrieved bytes:
  `c3ff5b8045c487339933f849c9f6b999f1e3fcf6e563538b40faffead21df342`)
- it is the right strain: `/strain="CEN.PK113-7D; CBS 8340"`, 6,171 gene features
- its locus tags are the **`SCEN_*` namespace** (18,207 `/locus_tag` lines, e.g.
  `SCEN_A00120`), not S288C systematic names

So provisioning the FILES needs no by-hand download. Depositing them is still not enough
to unblock the loader, for two measured reasons:

1. **Nothing reads it.** `READABLE_ASSEMBLY_SETS` (`torchcell/candidates/gates.py`) holds
   the bacterial sets plus `sgd_S288C_R64-4-1_20230830`, `peter2018_1011_assemblies` and
   `hale2024_322134S_variants`; the yeast side has `SCerevisiaeGenome` (SGD R64) and the
   Peter 2018 matrix, and no yeast class reads a GenBank assembly set. A deposited CEN.PK
   set would resolve but not ingest, so G2 would move from `blocked:provision` to
   `blocked:ingest`, not to pass.
2. **Two identifier spaces.** Zhang 2020 names its genes by their standard names (TRP2,
   ARO4), whose systematic names are S288C's, while the CEN.PK assembly's own tags are
   `SCEN_*`. Which namespace a CEN.PK genotype is written in is an owner decision, not a
   loader's.

**Owner question, in one line:** deposit `GCA_002571405.2` (CEN.PK113-7D) as a yeast
assembly set and add the yeast GenBank genome class that reads it, deciding whether a
CEN.PK genotype is keyed by `SCEN_*` tags or by S288C systematic names; until then the
Zhang 2020 loader stays unwritten and this note is its record. The schema it needs is
already served.

## 2026.10.12 - G2 is unblocked: the CEN.PK113-7D set is deposited and read

The owner's rule settles what the section above left as a question. 2026-10-12: "the data
is the data ... the genome is separate.. it doesn't gate anything", and on the namespace,
"again you are doing this sort of cross reference thing". So the genotype is keyed exactly
as the release keys it, no mapping to S288C systematic names happens inside a loader, and
a crosswalk, if ever wanted, is a derived artifact computed where it is used.

What landed, all additive genome-tier code (no schema change):

- `scripts/provision_yeast_genomes.py`, the yeast counterpart of
  `provision_bacterial_genomes.py`: fetches every member of an NCBI yeast assembly through
  the recorded `direct_url` retriever, checks each one against the md5 the assembly's own
  `md5checksums.txt` publishes, pins the sha256 of the bytes that passed in a
  `GenomeManifest`, deposits, and verifies. `--dry-run` stops before the deposit. There is
  no drift table to maintain: a changed upstream file fails its md5 check and stops the run.
- the set `scerevisiae_CENPK113-7D_ASM257140v2` (registry constant
  `SCEREVISIAE_CENPK113_7D`), deposited and verified from
  `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/571/405/GCA_002571405.2_ASM257140v2/`
  with five members, each md5-matched to that directory's listing:

  | member | role | sha256 (first 16) |
  |---|---|---|
  | `..._genomic.gbff.gz` | annotation | `c3ff5b8045c48733` |
  | `..._genomic.gff.gz` | annotation | `afc588ff9236f230` |
  | `..._genomic.fna.gz` | sequence | `8b3047dcfdc6ff80` |
  | `..._protein.faa.gz` | sequence | `c09aa2b4625b2e2e` |
  | `..._assembly_report.txt` | index | `0abbbdaaec3ead84` |

  No GO file: no GO release is keyed to this assembly's `SCEN_*` tags, and borrowing
  S288C's GO through a name match would attribute one strain's evidence to another
  strain's loci.
- `torchcell/sequence/genome/scerevisiae/ncbi.py`: `NcbiYeastGenome` (an
  `AnnotatedGenome`, GenBank-first) with `NcbiYeastAssembly`, `read_assembly_report`, and
  the concrete pair **`CenPk1137DGenome`** / `CenPk1137DGene`. It is a sibling of
  `BacterialGenome`, not a subclass: that class is correct for ONE replicon (its
  `fasta_chromosome` and `_read_sequences` refuse any other) and its GO route requires a
  GAF, while a yeast assembly has 17 replicons and no GAF. It reuses that module's file
  readers instead of widening its invariants, so no served bacterial class changes.
- `gates.HOSTS`: the `CEN.PK` token maps to the set, and `READABLE_ASSEMBLY_SETS` holds it
  because `CenPk1137DGenome` reads it.

MEASURED on the deposit (`tests/.../scerevisiae/test_ncbi.py`, the `data`-marked test):
18 replicons keyed from the assembly report's own table (16 chromosomes, the mitochondrion
at 0, the 2-micron plasmid at 17), **6,171 genes**, every tag in the `SCEN_*` namespace,
`go_annotations` empty. The locus-tag pattern was widened from the first guess on
measurement: 5,886 tags are `SCEN_<letter><5 digits>` and 285 are tRNA tags
(`SCEN_<letter>trna<n>[<amino-acid letter>]`, 4 of them with no trailing letter).

**The gate now reads admissible.** Re-run on this tree (the audit entry's composed G5 gap
is removed, since PR #1055 landed both classes):

```
zhangCombiningMechanisticMachine2020 -> admissible
  G1 pass  G2 pass: CEN.PK2-1C -> scerevisiae_CENPK113-7D_ASM257140v2
  G3 pass: 26 file(s) in library+raw, all intact
  G4 pass  G5 pass: BiosensorFluorescencePhenotype exists; compounds resolve
```

### Still to do: the loader itself

The loader is NOT written yet. What it will store, from the raw mirror (GitHub commit
`f95f7d10`, five files, every sha256 reproducing):

- 341 strain records: 299 library and control strains from `data/ice.csv.zip` (275 distinct
  five-promoter combinations, 3 replicate lines each) plus 42 ML-recommended strains from
  `data/recommendations.csv`, disjoint from the library.
- genotype: one `PromoterDrivenGeneIntegrationPerturbation` per re-integrated gene,
  `promoter_name` the released promoter and `integration_locus` the EasyClone site, keyed
  by the identifiers the release itself uses.
- phenotype: `BiosensorFluorescencePhenotype`, the specific GFP synthesis rate (MFI/h)
  derived from `data/edd.csv.zip`'s 75,026 GFP rows by the paper's own rule, with
  `sensed_metabolite` the typed L-tryptophan compound.
- plus the 6 HPLC tryptophan titers of `si/si1.md` (mg/L, SE, n = 4), which need a yeast
  product-titer pair: `ProductTiterExperimentReference.genome_reference` is an
  `AssemblyReferenceGenome` whose `assembly_set` vocabulary is bacterial-only, so that
  half is a schema decision of its own rather than part of this one.

The rule that turns the time series into a rate must be quoted from the paper before
anything is computed, which is the next step rather than a guess.
