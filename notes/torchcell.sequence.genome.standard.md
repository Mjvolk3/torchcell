---
id: p5nxp62jpjn9ucrs6vdkgtw
title: Standard
desc: ''
updated: 1795053661511
created: 1795053661511
---

## 2026.11.18 - Standard genomes by assembly set

`STANDARD_GENOMES` maps an assembly-set id to the genome class that reads it (five bacterial classes: E. coli K-12 BW25113, MG1655, W3110, B REL606, P. putida KT2440). `load_genome(assembly_set)` makes sure every set the class reads is in the genomes tier, downloading an absent one from `tc-data` only when `TC_DATA_URL` is set, and constructs the class. `genome_for(dataset)` reads the one `assembly_set` a dataset's reference index names and calls `load_genome`.

- A downloaded file is sha256-checked against its manifest row before the rename, and `manifest.json` is written last, so an interrupted fetch never looks deposited.
- The yeast classes are not in the table: `SCerevisiaeGenome` takes a `go_root` as well, so it needs its own row shape.
- Worked example for the four served E. coli datasets: `experiments/034-showcase-datasets/scripts/ecoli_genome_snippet.py`. Measured there (one run, GilaHyper, 2026-11-18): Campos 2018 names 3,664 genes, 3,573 current locus tags and 91 pseudogenes; Babu 2014 names 3,880 against MG1655, 17 not in the annotation; Fuhrer 2017 names 3,830 Keio JW numbers, 3,632 resolved through the gene synonym, 151 pseudogenes, 47 not in the annotation.

## 2026.11.19 - Automatic fetch, source first, mirror fallback

Owner policy (2026-11-19): a genome is fetched automatically and falls back to our mirror in the data lake. `load_genome` no longer needs `TC_DATA_URL`. The mirror's manifest is the pin authority; each file is downloaded from its recorded source URL and accepted only when size and sha256 equal the pin, otherwise it comes from the tc-data mirror (`torchcell.provenance.pinned_fetch.fetch_pinned`, written to be reused for network releases such as STRING). `fetch_report.json` beside the files records the origin of each one.

- The table now has seven rows: the five bacterial genomes plus S288C and CEN.PK113-7D.
- `go_release_2024-01-17` (`go.obo`, sha256 `ad2f0502...`) is a new set, deposited by `scripts/provision_go_release_20240117.py`. The dated release file is byte-identical to the copy cached on GilaHyper since 2024-03-15. `load_genome` places it at `<go_root>/go.obo` so `SCerevisiaeGenome` never reaches its download of the current GO file.
- Measured (one run from an empty directory on GilaHyper, 2026-11-19, 334 MB on disk): all seven genomes constructed; 49 of 63 files came from their source, 14 from the mirror. Of those 14, 8 are S288C release members with no recorded source URL, 5 were NCBI `HTTP 503` responses, and 1 (`109.P_putida_KT2440.goa`, EBI) has changed upstream (4,558,137 bytes against the pinned 4,502,711).
- Constructing a genome class directly still does not fetch; `registry.resolve` stays a pure path authority.
