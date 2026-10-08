---
id: 45t2hgmbtpk0968ascbwmrc
title: Test_bacterial_tier
desc: ''
updated: 1791423196112
created: 1791423196112
---

## 2026.10.07 - Cross-source checks of the four bacterial genomes

`tests/torchcell/sequence/genome/test_bacterial_tier.py` checks the GenBank-first ingest of
[[torchcell.sequence.genome.bacterial]] against the OTHER members of each assembly set,
which NCBI generates independently of the flat file, and against the GAF. The per-strain
modules ([[tests.torchcell.sequence.genome.ecoli.test_k12]],
[[tests.torchcell.sequence.genome.ecoli.test_rel606]],
[[tests.torchcell.sequence.genome.pputida.test_kt2440]]) pin what one genome reads from its
own route; this module asks whether two sources agree. One module-scoped fixture is
parameterized over MG1655, BW25113, REL606 and KT2440, built into temporary cache roots
with the network refusing; every test is `@pytest.mark.data` and runs with `--data` when
the four sets and the GO release are under `$DATA_ROOT/torchcell-genomes`.

Sources compared, every locus, not a sample:

- `_feature_table.txt.gz` (GCA): coordinates, strand, symbol, pseudogene class, protein
  accession and protein length of every locus agree with the `GenBankLocus`; the extra
  `with_protein` rows are exactly MG1655's ten isoform proteins.
- GFF3 in `data.db`: every gene and pseudogene row at the flat file's interval; the 18
  joined BW25113 pseudogenes pinned by tag, segments and symbol, each refused by name.
- `_genomic.fna.gz` against the CDS cut from the flat file's own sequence: equal for every
  coding gene but prfB (the CDS joins across the programmed frameshift, one base shorter,
  still translating to the 365 or 364 aa protein). Every CDS a whole number of codons;
  start codons pinned (336 GTG, 80 TTG, 4 ATT, 2 CTG in MG1655); translation equals
  `_protein.faa.gz` except the selenoproteins (fdnG, fdoG, fdhF; fdoG alone in KT2440).
- NCBI's own `_gene_ontology.gaf.gz` (GCF, keyed by `WP_` accession) crosswalked through
  the RefSeq GFF's CDS rows to the inline `Ontology_term` route BW25113 and REL606 store:
  the inline route is a superset on 2,164 of 2,175 (BW25113) and 2,184 of 2,195 (REL606)
  shared loci; the 11 loci where the GAF adds a term are pinned (trmD, truB, metA, ...).
  KT2440's GOA proteome file and NCBI's GAF agree on 186 of 3,091 shared loci: different
  annotations, as expected of UniProt vs PGAP.
- `ECOLI-uniprot.gaf.gz` column 2 against MG1655's `UniProtKB/Swiss-Prot` xrefs: 3,890 of
  3,890 agree; the GAF's column-3 symbol equals the GenBank symbol for only 3,775 (`lpdA`
  vs `lpd`), which is why the route joins on column 11.
- Windows at the replicon ends: upstream of thrL (190..255) or downstream of the last gene
  a window that runs off the end is refused by name unless `allow_undersize` clips it;
  the replicon is circular but a window never wraps.
- Name layers: duplicate symbols are AMBIGUOUS with gene-only candidates (nine IS symbols
  in BW25113, `metZ` tRNAs in REL606, `asd` in KT2440); `insI2` in MG1655 is shared by a
  gene and a pseudogene and resolves RENAMED to b1404; no two symbols differ only by case.
- The Carruthers 2025 chassis on KT2440: the eight deletion symbols, `phaC` and `glZ`
  RETIRED, PP_0815 CURRENT, and the stated 86,812 bp span holding 61 loci (57 genes).

### Two findings pinned as they stand

1. **Product strings wrapped in the flat file carry a spurious space.** NCBI wraps a long
   `/product` after a hyphen or a comma; Biopython's scanner joins the continuation with a
   space. 14 MG1655, 11 REL606 and 8 KT2440 coding loci therefore store
   `phospho-N-acetylmuramoyl-pentapeptide- transferase` where the GFF and the feature
   table both carry the unbroken string (`PRODUCT_WRAPPED`; b2024 wrapped twice). One
   REL606 product (`ECB_00636`) differs the other way: NCBI's derived files carry a stray
   space before a comma that the flat file lacks. The fix belongs in
   [[torchcell.sequence.genome.bacterial]], not here: take the product from the GFF the
   genome already loads, and refuse any disagreement that is not whitespace by name.
2. **The RefSeq inline GO route carries obsolete ids.** 49 (BW25113) and 50 (REL606)
   stored ids are obsolete in the pinned `go-basic.obo` (2026-07-26), 31 of each with a
   `replaced_by` successor; `remove_deprecated_go_terms` drops 96 and 95 locus-term pairs
   (emptying three and two loci) and does not remap to the successor. The GAF routes
   (MG1655, KT2440) carry none. Whether to remap through `replaced_by` is the owner's call.

The measurement scripts that produced every pin were run once and discarded; the test
is the record. Counts here are of the deposited bytes on 2026.10.07, and a change in any
member or in the parser fails by name.
