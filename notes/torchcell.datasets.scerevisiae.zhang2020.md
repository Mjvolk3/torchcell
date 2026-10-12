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
