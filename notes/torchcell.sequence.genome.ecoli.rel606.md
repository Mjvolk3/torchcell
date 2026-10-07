---
id: tjtxesgnvotfl6lle7j6fqb
title: Rel606
desc: ''
updated: 1791388874286
created: 1791388874286
---

## 2026.10.07 - E. coli B REL606 genome

The fourth bacterial reference, added so the Caglar 2017 multi-omics row
([[torchcell.datasets.ecoli.caglar2017]]) can pin its records. Module:
`torchcell/sequence/genome/ecoli/rel606.py`; shared bacterial layer:
[[torchcell.sequence.genome.bacterial]]; K-12 siblings:
[[torchcell.sequence.genome.ecoli.k12]]; tier: [[scripts.provision_bacterial_genomes]].
Tests: [[tests.torchcell.sequence.genome.ecoli.test_rel606]].

### Design

- `EcoliBREL606Genome(BacterialGenome[EcoliBREL606Gene])` binds `REL606_ASSEMBLY`:
  assembly set `ecoli_B_REL606_ASM1798v1`, GenBank `GCA_000017985.1_ASM1798v1`, RefSeq
  `GCF_000017985.1_ASM1798v1`, one circular replicon `CP000819.1` (chromosome key 1;
  any other seqid is refused), locus-tag pattern `ECB_[rt]?\d{5}`, default cache root
  `data/ecoli/rel606/genome`. It is built like `PPutidaKT2440Genome`: one class, one
  assembly, everything else from the shared layer.
- It is NOT a subclass of `EcoliK12Genome`. REL606 is an E. coli B strain: an `ECB_` tag
  is neither a b-number nor a `BW25113_` number, and the ECK crosswalk (an EcoCyc K-12
  accession) does not apply. `bacterial_genome("ecoli", "REL606")` returns this class
  under the same `ecoli_genome` loader parameter as the K-12 strains.
- `resolve_gene_name` has the K-12 classes' layer order (locus tag, `old_locus_tag`,
  RefSeq locus tag, gene symbol, `gene_synonym`, retired). The GenBank file carries no
  `gene_synonym` and no `old_locus_tag`, so on REL606 only the locus-tag, RefSeq-tag and
  symbol layers can resolve a name.

### GO source: the RefSeq GFF inline terms

Checked on 2026-10-07 for a locus-tag-keyed GAF anywhere scriptable:

- GO Consortium release 2026-08-05, `annotations/gaf/` index: the only E. coli files are
  `ECOLI-uniprot`, `ECOLI-mod` (K-12) and `ECO57-uniprot` (O157:H7). No E. coli B file.
- EBI GOA `proteomes/proteome2taxid` (sha256
  `5c85be0452ac2800c0ba33cbc1a1b7d098590d6786c917fa760587e0d5da151e`): no row for taxon
  413997, and none of its 9 E. coli strain proteomes (K-12, O157:H7, and seven other
  named isolates) is a B strain.
- NCBI PGAP `GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz` exists and is deposited
  (the provisioning script requires every listed GAF to be a member), but it is keyed on
  `WP_` proteins with an empty synonym column, so it names no locus tag.

So the GO route is `refseq_gff_ontology_term`, the BW25113 route: the
`Ontology_term` rows of `GCF_000017985.1_ASM1798v1_genomic.gff.gz` mapped to GenBank
tags through each RefSeq gene's `old_locus_tag`. All IEA (PGAP). The DAG is
`go-basic.obo` of `go_release_2026-08-05`.

### Measured on the deposited set (GilaHyper, 2026-10-07)

| measure | value |
|---|---|
| GenBank gene features | 4,383 (4,276 `ECB_NNNNN`, 85 `ECB_tNNNNN`, 22 `ECB_rNNNNN`) |
| gene set (non-pseudo) / pseudogenes / joined loci | 4,316 / 67 / 0 |
| coding CDS = proteins re-keyed to locus tags | 4,209 |
| CDS whose translation differs from the protein | 3 (`ECB_01432` fdnG, `ECB_03779` fdoG, `ECB_03951` fdhF: Sec) |
| `gene_synonym` / `old_locus_tag` values in the GenBank file | 0 / 0 |
| RefSeq genes / with `old_locus_tag` | 4,507 / 4,253 |
| `Ontology_term` rows / rows on RefSeq-only genes | 2,325 / 37 |
| GenBank loci reached / genes / pseudogenes | 2,253 / 2,232 / 21 |
| distinct GO terms over the gene set | 1,649 |
| terms obsolete in go-basic 2026-07-26 | 50 (all flagged `is_obsolete` in the OBO) |
| after `remove_deprecated_go_terms`: genes with GO / terms | 2,230 / 1,599 |
| symbols shared by two genes | 1 (`metZ`: `ECB_t00051`, `ECB_t00057`, resolves AMBIGUOUS) |
| build / reopen time | about 1.0 s / 0.6 s |

GO coverage is therefore 2,232 of 4,316 genes (51.7%) as stored, 2,230 (51.7%) after
obsolete terms are removed. Unlike MG1655 (whose GAF terms are all live in the pinned
release), REL606's RefSeq terms include 50 that go-basic 2026-07-26 has obsoleted.

Round trips (tier test): `ECB_00002`, `ECB_t00001`, `ECB_r00001` CURRENT; `thrA` and
`ECB_RS00010` RENAMED to `ECB_00002`; `ECB_99999`, `b0002`, `BW25113_0002`, `ECK0002`
RETIRED.

### Cache root

Built once with `overwrite=True` on 2026-10-07 at
`/scratch/projects/torchcell-scratch/data/ecoli/rel606/genome/data.db` (dev tree);
`database_untrusted_reason` returns None and `bacterial_genome("ecoli", "REL606")`
reopens it read-only (4,316 genes). The tier tests build into pytest temporary roots,
never this one.

### What it opens

- `torchcell.datamodels.schema`: `REL606` in `BacterialReferenceStrain`, set
  `ecoli_B_REL606_ASM1798v1` in `BacterialAssemblySet` and `BACTERIAL_ASSEMBLY_SETS`,
  accessions `GCA_000017985.1` / `GCF_000017985.1` in `ASSEMBLY_SET_ACCESSIONS`, and the
  namespace `ecoli_b_rel606_locus_tag` with pattern `^ECB_[rt]?\d{5}$` in
  `BacterialGeneNamespace` and `BACTERIAL_LOCUS_TAG_PATTERNS` (so every bacterial leaf
  admits an `ECB_` tag and requires the REL606 namespace on it).
  `python -m torchcell.provenance.schema_impact --base origin/main` printed
  `No schema contract changes vs origin/main.` (exit 0). That gate fingerprints the
  schema classes, whose annotation text names the Literal aliases rather than their
  members, so it does not see a Literal widening at all; the change is additive by
  construction (every previously valid value stays valid, no stored record serializes
  differently).
- `torchcell.datasets.bacteria_common`: REL606 in `HOST_STRAINS["ecoli"]`,
  `STRAIN_GENE_NAMESPACES` and `BACTERIAL_GENOME_CLASSES`; `bacterial_genome` has a
  REL606 overload; `BacterialStrainGenome` is the union `bacterial_genome` and the
  injector return.
- `torchcell.verification.runners`: `_ecoli_rel606_gene_set` (4,383 loci on the tier),
  and REL606 in `BACTERIAL_GENE_ASSEMBLIES`, so a record whose `genome_reference` names
  the REL606 set gets this universe and this genome.
- The Caglar 2017 gate: `strain_pin_finding()` is now pinnable, and Table S2's 4,196
  `ECB_` ids reconcile 4,196 CURRENT against this genome (measured, see
  [[torchcell.datasets.ecoli.caglar2017]]).
