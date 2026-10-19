---
id: b9v2xwhxhwdl6dvuy6hecye
title: Plan
desc: ''
updated: 1792444917852
created: 1792444917852
---

## 2026.10.19 - Organism similarity figure: is yeast more diverse than mammals?

### Question

The owner heard, from the natural-isolate literature, that there is more genetic difference
within yeast than within mammals, and wants (a) that claim checked and (b) a figure family
that places every organism torchcell serves or targets on common similarity axes: a tree or
cladogram, shared homologs, and enzyme (EC) repertoire. The organism set is the reference
yeast, the 1,011 natural isolates, our newly curated *Pichia kudriavzevii* SD108 genome,
*E. coli*, *P. putida*, the industrial hosts (10 established, 10 emerging, 4 CABBI
newcomers), and the mammals that make the point (human, great apes, macaque, mouse, rat,
hamster). Related: [[experiments.018-natural-isolate-genomics]] (per-ORF divergence of the
isolates against S288C, reused here), [[plan.genomes-tier.2026.09.14]] (where reference
sequence lives).

### What the literature actually says (checked 2026-10-19, zotlib index)

The claim is real but it is about the GENUS *Saccharomyces* (and, more loosely, the
subphylum Saccharomycotina), not about isolates of *S. cerevisiae*. Three statements, each
quoted from the mirror:

1. **Species of *Saccharomyces* are as far apart as mammals from each other or more; populations
   within a species sit between human/chimp and human/macaque.** Peris et al. 2023 (Nat Commun,
   `perisMacroevolutionaryDiversityTraitsGenomesModel2023`, Results, mirror line 36): "In
   Saccharomyces, levels of <85% of amino acid identity (AAI) in a set of core single-copy
   eukaryotic genes differentiated species, while population-level AAI values were higher
   (Fig. 3c). The lowest AAI value within a species was the comparison between the Asia B and EU
   populations of S. kudriavzevii, whose value was between the AAI values of the Homo
   sapiens/Pan troglodytes and Homo sapiens/Macaca mulatta comparisons." Their Methods (line
   189) define the measure we reuse: BUSCO `eukaryota_odb10` single-copy orthologs, MAFFT,
   mean amino acid identity per comparison, against Homo sapiens GRCh38, Pan troglodytes,
   Macaca mulatta, Mus musculus, Takifugu rubripes and Gallus gallus.
2. **The subphylum is as diverse as the animal kingdom.** Aranguiz et al. 2025
   (`aranguizMachineLearningRevealsGenesImpacting2025`, Introduction, line 13): "Yeasts of the
   subphylum Saccharomycotina display profound diversity at both the genetic and phenotypic
   levels. Their genetic diversity is comparable to that of the kingdom Animalia". The primary
   source for that sentence is Shen et al. 2018 (Cell), which is NOT in the Zotero index.
3. **Within *S. cerevisiae* the diversity is a few tenths of a percent at the nucleotide level,
   with Asia holding about half of it.** Peter et al. 2018 (`peterGenomeEvolutionAcross10112018`,
   line 70): "the genetic diversity of beer isolates is very high (π = 2.8 × 10^-3 on average)";
   Liti 2015 (`litiFascinatingSecretWildLifeBudding2015`, Abstract, line 5): "The recent
   discovery that Chinese isolates harbour almost twice as much genetic variation as isolates
   from the rest of the world combined". Our own 018 measurement: mean per-ORF divergence of
   an isolate from S288C is 0.42% (SNP only), 0.69% with indels
   ([[experiments.018-natural-isolate-genomics]], `per_strain_divergence_summary.parquet`).

So the honest version of the claim has three rungs, and the figure must show all three on one
axis: within *S. cerevisiae* (1,011 isolates), within the genus *Saccharomyces* (*S.
cerevisiae* vs *S. paradoxus* vs *S. eubayanus*), and within mammals (human vs chimp vs
macaque vs mouse vs rat vs hamster). Hypothesis (untested until the pipeline runs): the
isolates will sit near human/chimp, the genus will straddle human/mouse and beyond, and the
industrial yeasts (Saccharomycotina) will spread past human/chicken. The within-human
nucleotide diversity reference line (1000 Genomes) is a literature value; the implementer
sources and quotes it (DOI 10.1038/nature15393) and labels it as such in the caption.

Quoted files:

- `/bulk/zotero-library/perisMacroevolutionaryDiversityTraitsGenomesModel2023/paper.md`
- `/bulk/zotero-library/aranguizMachineLearningRevealsGenesImpacting2025/paper.md`
- `/bulk/zotero-library/peterGenomeEvolutionAcross10112018/paper.md`
- `/bulk/zotero-library/litiFascinatingSecretWildLifeBudding2015/paper.md`

### Organism set

Every organism gets one annotated reference proteome. Rule: the NCBI RefSeq reference
assembly for the species (`datasets summary genome taxon "<name>" --reference`), the named
strain when the species has an industrial type strain, resolved and RECORDED (accession,
assembly name, source URL, retrieval command, sha256 of the protein FASTA and GFF) in a
pydantic manifest `experiments/044-organism-similarity/results/organisms.json`. Organisms
already in the genomes tier are read from the tier through `torchcell.sequence.genome.registry.resolve`,
never re-downloaded. An organism with no annotated proteome at NCBI is kept on the taxonomy
cladogram and EXCLUDED from the proteome axes, with the absence stated in the manifest
(`proteome: null`, reason), never substituted by a relative. Mammalian proteomes are reduced
to one protein per gene (longest isoform, from the GFF `gene` parent) before any comparison,
otherwise isoform counts inflate the mammalian ortholog counts.

The justifications below are the owner's deep-research table (2026-10-19), condensed; the
"standing" column is that table's assessment, not ours.

| # | Organism, strain | Group | Why it is in the set |
|---|---|---|---|
| 1 | *Saccharomyces cerevisiae* S288C | reference | SGD R64-4-1 reference; tier `sgd_S288C_R64-4-1_20230830`, proteome `orf_trans_all_R64-4-1_20230830.fasta`. Established host #1: baker's yeast, ethanol, recombinant proteins; fed-batch is the standard biomass process. |
| 2 | *S. cerevisiae* 1,011 natural isolates | isolates | Peter 2018; tier `peter2018_1011_assemblies`. The within-species rung of the claim. |
| 3 | *Pichia kudriavzevii* (*Issatchenkia orientalis*) SD108 | emerging, CABBI | Our curated genome (private repo `Aurosish/I.orientalis_SD108_Genome_Reconstruction`, commit `8ca6a106`, 5 chromosomes + mito + rDNA, 5,144 nuclear protein-coding genes). Low-pH organic acids; succinate >100 g/L in bench fed-batch. |
| 4 | *P. kudriavzevii* CBS573 | emerging | The public reference (GCF_003054445.1) beside our SD108, so the figure shows how far our strain sits from the type strain. |
| 5 | *Escherichia coli* K-12 MG1655 | reference, established | Tier `ecoli_K12_MG1655_ASM584v2`. Established host #2: proteins, enzymes, amino acids. |
| 6 | *Pseudomonas putida* KT2440 | reference, emerging | Tier `pputida_KT2440_ASM756v2`. Emerging #13: aromatic valorization, PHA, beta-ketoadipate from glucose/xylose in fed-batch. |
| 7 | *Corynebacterium glutamicum* ATCC 13032 | established | Million-ton amino-acid fermentation (glutamate, lysine); robust aerobic fed-batch. |
| 8 | *Bacillus subtilis* 168 | established | Secreted industrial enzymes; the Gram-positive secretion workhorse. |
| 9 | *Bacillus licheniformis* DSM 13 (ATCC 14580) | established | Proteases and other secreted enzymes at industrial scale. |
| 10 | *Komagataella phaffii* GS115 (or CBS 7435) | established | "Pichia pastoris": recombinant enzymes, food proteins, biologics; glycerol/methanol fed-batch. |
| 11 | *Cricetulus griseus* (CHO) | established, mammal | Dominant mammalian platform for therapeutic proteins; suspension fed-batch is the manufacturing standard. Also the hamster point on the mammal axis. |
| 12 | *Homo sapiens* GRCh38 (HEK293) | established, mammal | Viral vectors (AAV, adenovirus) and proteins in suspension; the anchor of every mammalian comparison. |
| 13 | *Lactococcus lactis* IL1403 | established | Dairy starter biomass and nisin; long-established food-biotech organism. |
| 14 | *Aurantiochytrium limacinum* (Schizochytrium) | established | DHA oils; heterotrophic protist in sugar-fed reactors. Annotated proteome at NCBI is uncertain; apply the absence rule if none. |
| 15 | *Yarrowia lipolytica* CLIB122 | emerging, CABBI | Lipids, carotenoids, organic acids; EPA commercialized; CABBI conversion yeast. |
| 16 | *Rhodotorula toruloides* NP11 (or IFO0880) | emerging, CABBI | Microbial oils; 50 m^3 fed-batch scale-up reported 2025; CABBI conversion yeast. |
| 17 | *Cupriavidus necator* H16 | emerging | PHB; CO2/formate/H2-linked manufacturing. |
| 18 | *Kluyveromyces marxianus* DMKU3-1042 | emerging | Thermotolerant; lactose/whey valorization. |
| 19 | *Halomonas bluephagenesis* TD01 | emerging | PHA under nonsterile high-salt conditions; 5,000 L fed-batch in 2025. |
| 20 | *Ogataea polymorpha* NCYC 495 (or *O. parapolymorpha* DL-1) | emerging | Methylotrophic protein and vaccine-antigen host. |
| 21 | *Vibrio natriegens* ATCC 14048 | emerging | Fastest-growing chassis; ~60 g/L dry biomass in fed-batch. |
| 22 | *Methylorubrum extorquens* AM1 | emerging | Methanol-to-chemicals C1 chassis. |
| 23 | *Lipomyces starkeyi* NRRL Y-11557 | CABBI | Oleaginous; 130 g/L biomass and 88 g/L lipid in 5 L fed-batch; CABBI systems-biology organism. |
| 24 | *Zygosaccharomyces bailii* CLIB 213 | CABBI | Acid- and preservative-tolerant; CABBI CRISPR strain made 35 g/L lactate. |
| 25 | *Zygosaccharomyces rouxii* CBS 732 | CABBI | Osmotolerant; D-arabitol 152.8 g/L in fed-batch; CABBI sequenced NRRL Y-64007. |
| 26 | *Lipomyces tetrasporus* | CABBI | CABBI sequenced NRRL Y-64009; lipid accumulation on hydrolysates. NCBI annotation uncertain; absence rule applies. |
| 27 | *Saccharomyces paradoxus* CBS432 | genus | The sister species: the genus rung of the claim. |
| 28 | *Saccharomyces eubayanus* FM1318 | genus | The far end of the genus (lager parent). |
| 29 | *Pan troglodytes* | mammal | Human/chimp is the floor of mammalian divergence (Peris's lower anchor). |
| 30 | *Gorilla gorilla* | mammal | Great ape. |
| 31 | *Pongo abelii* | mammal | Great ape, the most distant. |
| 32 | *Macaca mulatta* | mammal | Peris's second anchor. |
| 33 | *Mus musculus* C57BL/6J | mammal | The model rodent; human/mouse is the classic ~85% protein identity pair. |
| 34 | *Rattus norvegicus* | mammal | Model rodent. |
| 35 | *Gallus gallus* | outgroup | Peris's amniote outgroup; bounds "within mammals" from outside. |

Thirty-four reference proteomes plus the isolate panel. Established = rows 1, 5, 7-14;
emerging = rows 3, 6, 15-22; CABBI = rows 3, 15, 16, 23-26; mammals = rows 11, 12, 29-34.

### Similarity axes (standard, cheap first)

All computed by committed scripts under `experiments/044-organism-similarity/scripts/`,
run from the repo root with the `orgsim` env
(`/scratch/projects/torchcell-scratch/envs/orgsim/bin/python`; mmseqs2, diamond, NCBI
`datasets`, eggNOG-mapper, mafft, FastTree, IQ-TREE, rapidnj, ete3, biopython). Large
intermediates go to `$DATA_ROOT/data/organism_similarity/`; the eggNOG database to
`$DATA_ROOT/data/eggnog/`. Results tables go to `experiments/044-organism-similarity/results/`.

1. **Homolog overlap and amino acid identity (AAI).** One `mmseqs easy-search` of the
   concatenated 34-proteome database against itself (`-s 5.7`, `-e 1e-5`, `--cov-mode 0 -c
   0.5`, `--max-seqs 4000`, 96 threads), then reciprocal best hits (RBH, by bitscore) per
   ordered pair in Python. Per pair record: `n_rbh`, `ortholog_fraction = n_rbh / min(n_i,
   n_j)`, `ortholog_jaccard = n_rbh / (n_i + n_j - n_rbh)`, `aai_mean` and `aai_median` =
   identity over the RBH pairs. This is the CompareM definition of AAI and the quantity
   Peris plotted (theirs over BUSCO orthologs; we also compute the BUSCO-restricted variant
   as a check, see 4). Output `results/pairwise_similarity.csv`.
2. **EC repertoire.** eggNOG-mapper 2 (`emapper.py -m diamond --cpu 64`) on every proteome,
   one run per organism, uniform annotation so a poorly curated yeast is not penalized against
   human. From the `EC` column build per-organism EC sets and count vectors at level 4
   (`1.1.1.1`) and level 3 (`1.1.1.-`). Per pair: `ec4_jaccard`, `ec3_jaccard`, and Bray-Curtis
   on the level-4 count vectors. KEGG KO and COG categories come for free; store them. Output
   `results/ec_profiles.parquet`, pairwise columns appended to `pairwise_similarity.csv`.
   CLEAN (the owner's later idea) replaces this step's annotator, nothing else.
3. **Trees.** (a) The NCBI taxonomy cladogram over the 34 taxids (ete3 `NCBITaxa`),
   topology only, the ordering axis of every heatmap. (b) A neighbor-joining tree from AAI
   distance (`d = -ln(aai)` over RBHs, Biopython NJ), rooted on the bacteria, drawn beside the
   cladogram so disagreements are visible. (c) The 1,011-isolate tree: rapidnj on Peter's
   released `1011DistanceMatrixBasedOnSNPs.tab.gz` (`$DATA_ROOT/data/peter2018/`, pin its
   sha256), tips colored by Peter's clade assignment from SI Table 1 (mirror
   `torchcell-library/peterGenomeEvolution10112018/si/si1.md`), and `SACE_*` isolates
   that Peter marks as the reference flagged.
4. **The claim figure (the headline).** One axis, protein identity over shared orthologs, as
   a strip plot of pair groups: within *S. cerevisiae* (each isolate vs S288C; reuse 018's
   per-strain divergence and ADD a protein-level number by translating the isolate's
   reference-inferred CDS from `allReferenceGenesWithSNPsAndIndelsInferred.tar.gz` in the
   tier and scoring identity against the S288C protein, mean over ORFs), within the genus
   (rows 1, 27, 28 pairwise), within mammals (rows 11, 12, 29-34 pairwise), mammal vs chicken,
   within Saccharomycotina (every yeast pair), within bacteria, yeast vs bacteria. Human/chimp,
   human/macaque, human/mouse and human/chicken are labeled points. Re-derive Peris's
   BUSCO-restricted AAI for the same pairs (compleasm `eukaryota_odb10` is in the env) and
   state in the caption whether the two definitions agree.

### Figures

Saved to `ASSET_IMAGES_DIR/044-organism-similarity/` as PNG and true-size SVG
(`torchcell.utils.savefig_true_size_svg`), `PANEL_WIDTHS_MM`, Arial 6 pt, boxed axes,
`PLOT_PALETTE` with color = organism GROUP (established, emerging, CABBI, mammal, genus,
isolates), timestamped filenames until approved, embedded as SVG in
[[experiments.044-organism-similarity]]. Every figure has a numbered caption that states the
scoring rule and the n.

- Fig 1 (full width): the claim strip plot (axis 4) with the within-human literature line.
- Fig 2 (full width): taxonomy cladogram on the left, three heatmaps in cladogram order:
  AAI, ortholog Jaccard, EC-4 Jaccard.
- Fig 3 (half): AAI-distance NJ tree vs cladogram (tanglegram or side by side).
- Fig 4 (half): the 1,011-isolate tree colored by clade, with S288C marked.
- Fig 5 (half): ortholog fraction vs AAI per pair, group-colored.

### Division of work

- **Agent A, separate worktree `feat/pkudriavzevii-sd108-genome`:** deposit the SD108
  release into the genomes tier as `pkudriavzevii_SD108_v1` (registry constant
  `PKUDRIAVZEVII_SD108`), then the genome class
  `torchcell/sequence/genome/pkudriavzevii/sd108.py` reading FASTA + GFF3 (+ `proteins.faa`,
  `cds.fna`, `gene_xref.tsv`), GO stated absent as CEN.PK does, tests against the tier,
  paired notes. PR of its own.
- **Agent B, this worktree:** the 044 pipeline, scripts 1-4, figures, the experiment note,
  the organisms manifest. Reads SD108's proteome from the clone at
  `/scratch/projects/torchcell-scratch/tmp/044-inputs/sd108-repo/Assembly_Build/final/annotation/proteins.faa`
  (md5 `214f96621ada17ac32f594fc274ecf70` per the repo's MD5SUMS; the manifest records the
  sha256) and switches to `resolve(PKUDRIAVZEVII_SD108, "proteins.faa")` once Agent A lands.

### Gotchas

- Mammalian isoforms (above). Human RefSeq has ~120k proteins for ~20k genes.
- `datasets download genome accession ... --include protein,gff3` zips; hash the extracted
  members, record the zip URL and the `datasets` version.
- eggNOG data download is ~10 GB (`download_eggnog_data.py -y --data_dir`); run once,
  record the eggNOG DB version in the manifest.
- The Peter distance matrix is a 1011 x 1011 table whose first column is the isolate code
  (`STD` header); values are SNP-based distances on Peter's scale, not per-site π. Use it
  for the tree only; the per-site numbers come from 018.
- Do not pass `bbox_inches="tight"` on fixed-width panels. Palette rule: color is the
  organism group, never the method.
