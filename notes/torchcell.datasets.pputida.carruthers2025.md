---
id: sb9mko8qpj34pjl1yy97i33
title: Carruthers2025
desc: ''
updated: 1791383735616
created: 1791383735616
---

## 2026.10.07 - The first P. putida loader: CRISPRi isoprenol titer and its proteome panel

`torchcell/datasets/pputida/carruthers2025.py` serves rank 6 of the fifty bacterial rows
([[plan.bacteria-ontology-genome]] section 4): Carruthers et al. 2025, "Automation and
machine learning drive rapid optimization of isoprenol production in Pseudomonas putida"
(Nat Commun, doi:10.1038/s41467-025-66304-8, PMID 41390487, PMC12748988), citation key
`carruthersAutomationMachineLearning2025`.

Two dataset classes, because `ExperimentDataset.transform_item` validates against ONE
`experiment_class`:

| class | family | records | references | gene set |
|---|---|---|---|---|
| `IsoprenolTiterCarruthers2025Dataset` | `ProductTiterExperiment` | 465 | 7 (one per DBTL cycle) | 126 |
| `ProteomeCarruthers2025Dataset` | `BacterialProteinAbundanceExperiment` | 19 | 1 | 20 |

The titer gene set is 121 guide targets plus the 5 pIY670 pathway tokens; the proteome
gene set is 14 off-target guide targets plus `PP_0815` plus the 5 tokens.

### What each record is

`genome_reference` is `assembly_reference("KT2440", background=chassis_background(genome))`:
species *Pseudomonas putida*, strain `IY1449b`, assembly set `pputida_KT2440_ASM756v2`,
GenBank accession `GCA_000007565.2`. The pin survives serialization because both
bacterial reference classes re-annotate `genome_reference` to `AssemblyReferenceGenome`
(the caveat in [[torchcell.datamodels.bacterial-perturbation-ontology]]), which a
data-gated test asserts on a real dump.

Everything above the chassis is a perturbation in `Genotype`:

- five `HeterologousPathwayPerturbation`, one per pIY670 gene, so `IY1449b` plus the
  pathway IS the production strain `IY1452b`;
- one `BacterialCrisprInterferencePerturbation` per guide target, carrying
  `CrisprConstruct(effector="dCas9", guide_sequence=None, n_guides=1)`;
- in the proteome family only, a `BacterialDeletionPerturbation` for the chromosomal
  `PP_0815` knockout its panel was built in.

### Sourcing: the chassis genotype table

Two mirrored statements of `IY1449b` disagree, and both are kept as `provenance` quotes
on the background rather than one being preferred.

- Methods, `paper.md` sha256 `ca9a8a2593d2ae3ab3bacfb767e797ece2bdaa0228a4f798f1c38e1af73ef88d`:
  "The selected chassis strain, P. putida IY1449b, has the in-frame deletions ΔphaABC,
  ΔmvaB, ΔhbdH, and 4,538,575Δ86,812 (Δzwf, ΔglZ, and ΔliuC) for improved isoprenol
  titers"
- Fig. 2 caption, same bytes: "Validated sgRNA arrays were then dispensed with
  electrocompetent P. putida IY1449b (ΔphaABC, ΔmvaB, ΔhbdH, ΔldhA, ΔzwfB, ΔgntZ, and
  ΔliuC) cells harboring pIY670 again using the ECHO 550."
- Supplementary Table 2, `si/si1.md` sha256 `2f83cfecc539e6607e33c51c420fb0be95c335a494cba6ef485eb5cb627ea0a5`,
  row `IY1449b` / `JBx_273364`: "KT2440 ΔphaABC, ΔmvaB, ∆hbdH, 4,538,575 Δ86,812". This
  string is stored verbatim as `genotype_statement`.

What reconciles the two lists is measured on `GCA_000007565.2`, not chosen. The stated
span 4,538,575 plus 86,812 bp is 4,538,575..4,625,386, and it contains `PP_4042` (zwfB,
4,554,991..4,556,496), `PP_4043` (gntZ, 4,556,493..4,557,476) and `PP_4066` (liuC,
4,590,930..4,591,745) entirely, while the bare symbol `zwf` resolves to `PP_5351` at
6,099,177, well outside it. So the Methods' `zwf` is the caption's `zwfB` and its `glZ`
is `gntZ`. `ldhA` resolves to `PP_1649` at 1,840,974, also outside the span, which is
consistent with the Methods omitting it and the caption listing it separately; it is
typed on the caption's authority and that is said here rather than hidden.

Eight typed `BacterialBackgroundAllele` entries, each `full_deletion` with
`functional=False`, each locus resolved by the genome from the SOURCE's own symbol:

| symbol | locus | designation | deleted_span | named by |
|---|---|---|---|---|
| phaA | `PP_5003` | ΔphaABC | none | both lists |
| phaB | `PP_5004` | ΔphaABC | none | both lists |
| mvaB | `PP_3540` | ΔmvaB | none | both lists |
| hbdH | `PP_3073` | ΔhbdH | none | both lists |
| ldhA | `PP_1649` | ΔldhA | none | Fig. 2 caption only |
| zwfB | `PP_4042` | ΔzwfB | 4,538,575..4,625,386 | Fig. 2 caption (Methods' `zwf`) |
| gntZ | `PP_4043` | ΔgntZ | 4,538,575..4,625,386 | Fig. 2 caption (Methods' `glZ`) |
| liuC | `PP_4066` | ΔliuC | 4,538,575..4,625,386 | both lists |

### The two gaps in that background, and why neither is a ProvenanceGap

- **The span removes 57 annotated genes, and the source names three.** Measured: 57 gene
  features of `GCA_000007565.2` lie entirely inside 4,538,575..4,625,386. The other 54
  removed loci carry no allele record.
- **`phaC` is not a gene symbol of this assembly.** `phaA` and `phaB` resolve;
  `resolve_gene_name("phaC")` returns `retired`. The third gene of ΔphaABC therefore gets
  no locus, and borrowing a neighboring tag would be exactly the cross-strain inference
  decision D9 refuses.

Neither is typed, because `ProvenanceGap` must name a field that is `None` and these are
missing ROWS, not missing fields. Both survive verbatim in `genotype_statement`, and the
loader asserts at build time that `phaC` still fails to resolve, so the day the
annotation gains the symbol the build stops and says so.

### Sourcing: the pIY670 pathway, corroborated by the released proteomics

The part composition is a verbatim cell of Supplementary Data 3 (`si/si6.xlsx` sha256
`e6704d6176d61c76f12243e8248bcff0767d7d57c6a0737da07d5574072c4759`), plasmid `pIY670`
`JBx_264945`: `pRK2-Kan-araC-PBAD-MvaSEf-MvaEEf-TrpoH-Ptrc1-O-MKMm-PMDScHKQ-AphA`.

`source_organism` is NOT read off the `Ef` / `Mm` / `Sc` suffixes. Every part has a
measured UniProt entry in the Source Data proteome sheet, and the organism mnemonic in
`Protein.Names` agrees with its suffix:

| stored identifier | `Protein` symbol | `Protein.Group` | `Protein.Names` | `source_organism` | promoter |
|---|---|---|---|---|---|
| `MvaSEf` | Mvas | Q9FD71 | `HMGCS_ENTFL` | Enterococcus faecalis | PBAD |
| `MvaEEf` | Mvae | Q9FD70 | `Q9FD70_ENTFL` | Enterococcus faecalis | PBAD |
| `MKMm` | Mvk | Q8PW39 | `Q8PW39_METMA` | Methanosarcina mazei | Ptrc1-O |
| `PMDScHKQ` | Mvd1 | P32377 | `MVD1_YEAST` | Saccharomyces cerevisiae | Ptrc1-O |
| `AphA` | Apha | P0AE22 | `APHA_ECOLI` | Escherichia coli | Ptrc1-O |

`MVD1_YEAST` independently agrees with the Methods' own statement, "The base pathway
(Fig. 1a) is an engineered mevalonate (MVA) pathway with a promiscuous mevalonate
decarboxylase (PMD\*) from S. cerevisiae ... thereby bypassing isopentenyl diphosphate
(IPP-Bypass)", which is also where `variant="HKQ"` and the pathway name come from. The
mnemonic to species expansion is UniProt's controlled vocabulary.

`systematic_gene_name` is the VERBATIM plasmid part token, because that is the identifier
a mirrored file states; an organism-qualified name such as `Efa:mvaS` would be a rewrite.
The loader asserts each token is a substring of the quoted description, so a quote edit
cannot silently orphan an identifier.

The promoter assignment is READ from the part order, not stated in prose: `araC-PBAD`
precedes MvaSEf and MvaEEf, the `TrpoH` terminator closes that operon, and `Ptrc1-O`
opens the next. The arabinose induction corroborates PBAD independently ("Isoprenol
pathway genes were induced after 8 h by the addition of L-arabinose").

`copy_number` is 1.0, meaning one cassette copy per gene. The pRK2 plasmid's copy number
is not stated anywhere in the release.

### Sourcing: the titer column, n_samples and the uncertainty type

The loader consumes the Source Data workbook, `41467_2025_66304_MOESM9_ESM.xlsx`, sha256
`1b3a7ab5f165386ba1c11e8873c397e7b03f5189274c0a423c5c60dd3616c1c7`, sheet **`Figure 4b`**,
column **`isoprenoli titer (mg/L)`** (the typo is the source's), with `Line Name`, `cycle`,
`is_control` and `pass filter?`.

- **`n_samples`**, Methods "Statistics & reproducibility", verbatim: "All strains were
  cultured as biological triplicates $\left( n = 3 \right)$ ." So `sample_unit` is
  `biological_replicate`. Each record stores the replicate count it actually has, which
  is 3 for 458 strains and 6 for seven of them.
- **Uncertainty TYPE**, same section, verbatim: "With the exception of the
  box-and-whisker plot in Fig. 4d, all error bars represent standard deviation." So
  `titer_uncertainty_type = sample_sd` and `titer_se = SD / sqrt(n)`. The loader reads no
  value from Fig. 4d.
- **Quantification**, verbatim: "Isoprenol from BioLector experiments was detected using
  gas chromatography-flame ionization detection (GC-FID; Agilent Technologies, Santa
  Clara, CA)." stored as `quantification_method`.
- **Yield and productivity** are four typed `ProvenanceGap`s with reason
  `not_reported_by_primary`: the campaign released neither.

### The unit decision: mg/L stored verbatim as ug/mL

`ConcentrationUnit` has no `mg/L` member (`M`, `mM`, `uM`, `nM`, `percent_v/v`,
`percent_w/v`, `ug/mL`, `g/L`, `pH`). 1 mg/L is exactly 1 ug/mL, so the released number
is stored unchanged under the numerically identical unit rather than divided by 1000.
Adding `mg_per_l` to `ConcentrationUnit` would read better, but that enum is in every
served dataset's closure, so it is a rebuild decision and is raised in the PR instead.

### Identifier reconciliation, both families

**Titer family, 121 distinct guide targets:**

| status | count |
|---|---|
| current | 121 |
| renamed | 0 |
| non_gene_feature | 0 |
| retired | 0 |
| ambiguous | 0 |

Layers: locus tag 121, old locus tag 0, RefSeq locus tag 0, gene symbol 0, gene synonym
0, not found 0. Remapped 0, kept on collision 0, outside namespace 0. Resolved fraction
1.000, and `MIN_RESOLVED_FRACTION = 1.0` so a future annotation change stops the build
rather than quietly degrading.

**Proteome family, 1,501 distinct `Protein` keys:**

| status | count |
|---|---|
| current | 497 |
| renamed | 928 |
| non_gene_feature | 0 |
| retired | 75 |
| ambiguous | 1 |

Layers: locus tag 497, gene symbol 929, not found 75. Remapped 1,425, kept on collision
0, outside namespace 76, ambiguous `Asd` with candidates (`PP_1989`, `PP_1992`). Resolved
fraction 0.9494, and `MIN_RESOLVED_FRACTION = 0.94`.

The sheet's `Protein` column mixes `PP_` tags with title-cased UniProt gene symbols
because the search database was built that way. Methods, verbatim: "The database used in
the DIA-NN search (library-free mode) included the latest P. putida KT2440 Uniprot
proteome FASTA sequences in addition to the protein sequences of heterologous proteins
and common proteomic contaminants." Measured organism mnemonics in the released sheet:
PSEPK 89,460 rows, HUMAN 240, ENTFL 120, and 60 rows each of KLEPN, PIG, ECOLI, YEAST,
METMA and STRP1.

### Drops

**Titer family: zero.** Every one of the 1,416 released non-control cultures is in a
record. The 90 control cultures are the per-cycle `phenotype_reference`, not records.

The authors' per-strain CRISPRi proteomics filter is carried in
`preprocess/pass_filter.csv` and is deliberately NOT on the record: it reports whether
the DESIGNED knockdown was REALIZED, and no certainty axis exists to type that on a
perturbation (memory `designed-vs-realized-perturbation-material-entity`). The paper is
explicit that the filter shaped only model training: "Data used to train the active
learning model was filtered according to the method above; however, no data was excluded
in our analysis." Measured: 208 of the 465 strains pass, by cycle 68/121, 29/57, 12/59,
20/54, 22/57, 42/58, 15/59.

**Proteome family: no sample is dropped; 77 of 1,501 protein KEYS are**, listed with
their accessions and descriptions in `preprocess/dropped_protein_keys.csv`, leaving 1,424
in every record's abundance map:

- 76 keys that are no locus of the pinned assembly: the pathway proteins (`Mvas`, `Mvae`,
  `Mvk`, `Mvd1`), the `Cas9` effector, the `Neo` resistance marker, the four human
  keratins `Krt1`, `Krt2`, `Krt9`, `Krt10`, and about 65 host proteins whose UniProt
  symbol spelling the GenBank annotation does not carry (`Acsa1`, `Aroa`, `Bama`, ...).
- `Apha`, which DOES resolve but which the released sheet files under one symbol for two
  distinct protein groups: the heterologous `APHA_ECOLI` (P0AE22) and a native
  `Q88C43_PSEPK`. The key names two proteins, so its abundance cannot be attributed.
  `Asd` (Q88LE2 / Q88LE4) is the other merged key and is also in the 76.

A released Top3 value of 0 is kept verbatim. The companion percent-abundance column
floors those cells at 1e-05; the loader imputes nothing.

### Record counts, and why they differ from the abstract's

The paper reports "472 unique strains (125 single perturbations and 347 combinations) in
triplicate". 472 is the 1,416 non-control cultures divided by three. Grouping by
`(construct, DBTL cycle)` gives **465** strains, because seven carry SIX replicates
(R1-R6) rather than three: `PP_0812`, `PP_0813`, `PP_4678`, `PP_4679` in DBTL0 and
`PP_0814_PP_4192`, `PP_0814_PP_4862`, `PP_2137_PP_4189` in DBTL1. 465 plus 7 is 472 only
under the divide-by-three reading. Guide-count histogram over the 465: 123 one-guide, 136
two, 168 three, 38 four.

Two DBTL0 constructs are named `PP_1607_NT1` and `PP_4194_NT2`: one target plus a
non-targeting filler guide occupying an array position. The filler perturbs no gene, so
those records carry one CRISPRi perturbation and the token is recorded in
`preprocess/pass_filter.csv`.

### Build-time cross-source assertions, all measured to hold

1. `Figure 1D` and `Figure 4b` are the same 1,506 rows in the same order with
   bit-identical titers. The loader refuses any disagreement.
2. `Figure 3a`'s released per-target mean equals the mean over that target's DBTL0
   replicates for all 119 joinable targets, worst |diff| 3.3e-8 mg/L.
3. Supplementary Data 1's `Mean isoprenol titer (mg/L)` agrees for all 118 of its
   name-joinable targets within its own two-decimal rounding, worst |diff| 0.005 mg/L.
4. `Figure 2c`'s control titers equal `Figure 4b`'s control rows exactly in DBTL1 to
   DBTL6 and to 3.3e-6 mg/L in DBTL0, which the two sheets export at different precision.
5. Control counts match the Fig. 2 caption, verbatim: "In DBTL0, the control strain was
   cultured across plates $( n = 1 8 )$ while subsequent cycles had three control strains
   per plate $( n = 1 2 )$ )." Measured 18, 12, 12, 12, 12, 12, 12.

**One cross-source disagreement, kept as a finding:** `PP_3365` is a DBTL0 guide target in
the Source Data but is absent from Supplementary Data 1's 120-row target table, so that
table covers 120 of the 121 targets screened.

**A second, smaller one:** Supplementary Data 1 marks 65 of its 120 targets as passing
the DBTL0 filter (49 passed and unused after DBTL2/3, 16 passed and used), while the
Methods state "Filtering resulted in 67 validated sgRNA targets." The two numbers are
reported as found and nothing is reconciled.

### Control reproducibility, measured

`preprocess/cycle_controls.csv`:

| cycle | n | mean (ug/mL) | SD | CV % |
|---|---|---|---|---|
| DBTL0 | 18 | 169.766 | 19.851 | 11.69 |
| DBTL1 | 12 | 154.402 | 10.195 | 6.60 |
| DBTL2 | 12 | 156.733 | 15.693 | 10.01 |
| DBTL3 | 12 | 168.831 | 16.776 | 9.94 |
| DBTL4 | 12 | 152.855 | 7.047 | 4.61 |
| DBTL5 | 12 | 163.020 | 9.027 | 5.54 |
| DBTL6 | 12 | 155.195 | 8.900 | 5.73 |

The paper claims "our controls showed only a $5 \mathrm { - } 1 0 \%$ coefficient of
variation of isoprenol titer across all cycles". That band holds for DBTL1 to DBTL6
(4.61 to 10.01 percent); DBTL0's 18 control cultures give 11.69 percent.

### Environment

The shared `media.M9_NREL_CARRUTHERS2025` object (see [[torchcell.datamodels.media]],
2026.10.07), 24 C, 48 h, aerobic, with L-arabinose at 2 g/L as a
`SmallMoleculePerturbation`. Quotes: "Cultures were grown at $2 4 ^ { \circ } \mathrm { C
}$ and shaken at 1000 RPM without humidity control as batch experiments." and "Isoprenol
pathway genes were induced after 8 h by the addition of L-arabinose to a final
concentration of ${ 2 \mathrm { g } } / { \mathrm { L } }$ . Following $4 8 \mathrm { h }$
of production ...".

**It is a plain `Environment`, not a `CultureEnvironment`, and that is a schema finding
rather than a preference.** `Experiment.environment` is annotated `Environment` and
pydantic serializes by the declared type, so a `CultureEnvironment` placed there keeps
`culture_format` as a python attribute and dumps WITHOUT it, with no error anywhere. A
test pins both halves of that behavior. The vessel ("48-well BioLector flower plate"),
the 1.5 mL working volume, the 1000 RPM shaking and the 48 h endpoint are therefore
recorded in the module's `CULTURE_FORMAT` sourced value and here, and the fix is to
narrow `ProductTiterExperiment.environment` (raised in the PR, not taken in this branch).

Two more environment facts are recorded here rather than typed, because `ProvenanceGap`
requires the gapped field to be `None` and `perturbations` is set:

- the production culture's kanamycin and gentamicin concentrations are not stated; the
  Methods give 50 ug/mL kanamycin and 10 ug/mL gentamicin for the LB and passaging steps
  only;
- `aerobicity="aerobic"` is read from a flower plate shaken at 1000 RPM under a
  gas-permeable seal; the source never uses the word.

`isoprenol` has no row in the committed `compound_identity_table.json`, so
`resolved_compound("isoprenol")` returns the canonical name with a typed
`ProvenanceGap` on `inchikey`. Curating the row needs a PubChem call against the
committed input lists and is a human act; it is raised in the PR.

### The proteome arm: what is loaded, and what is blocked

The 472-strain campaign proteome is NOT in the mirror. It lives in seven PRIDE projects of
raw DIA files, one per cycle, all enumerated in the raw mirror's `si_data_sources`:
PXD063733 (DBTL0), PXD063737 (DBTL1), PXD063738 (DBTL2), PXD063740 (DBTL3), PXD063743
(DBTL4), PXD063744 (DBTL5), PXD063746 (DBTL6). No loader consumes raw spectra, so none is
deposited.

The processed campaign proteomics is in Dryad `10.5061/dryad.gtht76hzh`:
`CRISPRi_automation_Pputida_proteomic_Top3_peptide_quantification_method_data.csv`
(29,700,365 B), `CRISPRi_automation_Pputida_proteomic_metadata.csv` (333,287 B) and
`README.md` (5,768 B). **It is not scriptable.** Measured 2026-10-07: `datadryad.org`
serves an Anubis JavaScript proof-of-work challenge on the public
`/downloads/file_stream/<id>` route (HTTP 200, challenge page) and HTTP 401 "Unauthorized,
must have current bearer token" on `/api/v2/files/<id>/download`. The file listing itself
(`/api/v2/versions/383277/files`) IS readable, which is how the three names and sizes
above were measured. Manual recipe, recorded in the mirror's `si_expected`: open
`https://doi.org/10.5061/dryad.gtht76hzh` in a browser, solve the challenge, use
"Download dataset", then deposit the three files under `data/dryad/` with
`RetrievalMethod.manual_browser` and the sha256 of the bytes that arrive.

What IS loaded is the only per-protein, per-replicate abundance matrix the Source Data
carries: sheet `Supplementary Figure 13abc`, 90,180 cells over 20 samples, 3 replicates
and 1,501 protein groups. Column consumed: **`Top_3pep_counts_mean`**, with
`measurement_type="dia_nn_top3_peptide_signal_mean"`. Methods, verbatim: "The Top3 method,
which is the average MS signal response of the three most intense tryptic peptides of each
identified protein, was used to plot the quantity of targeted proteins in the samples".

The panel is the PP_0815 off-target study: 18 samples over the 14 candidate targets of
Supplementary Table 1 plus `JBEI_PP_0815_Target_48hr` and `JBEI_PP_0815_NT_48hr`. The
non-targeting control is the `phenotype_reference`; the other 19 are records.
`PP_0977` appears three times, `PP_1638` three times and `PP_3416` once with a plate
suffix, because "These strains were cultured three times owing to poor transformation
efficiency." Each is an independent culture of the same genotype and is kept as its own
record, never averaged.

`n_samples` for this family, Supplementary Fig. 13 caption, verbatim: "All strains were
cultured in triplicate $( \mathsf { n } = 3 )$ and error bars represent standard
deviation." The loader asserts every (sample, protein) cell has exactly three replicates.

The panel's background is the chromosomal `PP_0815` knockout. The main text writes it
"IY1452b ΔPP_0815" and the SI caption writes the same strain as "sgRNA expressed in
IY1449b ΔPP_0815"; both mean `IY1449b ΔPP_0815` carrying pIY670, since Supplementary
Fig. 13d reports isoprenol titers for these strains.

### Raw mirror

`$DATA_ROOT/torchcell-raw/carruthersAutomationMachineLearning2025/` with a
`torchcell.literature.manifest.Manifest`:

| path | role | bytes | sha256 |
|---|---|---|---|
| `si/41467_2025_66304_MOESM9_ESM.xlsx` | `si_data` | 10,916,318 | `1b3a7ab5f165386ba1c11e8873c397e7b03f5189274c0a423c5c60dd3616c1c7` |
| `si/41467_2025_66304_MOESM4_ESM.xlsx` | `si_data` | 20,093 | `236c8dc5d3b18b7a459f69a6812efa534bba13cd49e20e1a89a01b7371d1e54b` |

Both carry a `RetrievalRecord` with `method=pmc_cloud`,
`retriever="torchcell.literature.retrieve.pmc_cloud_object"` and
`params={"key": "PMC12748988.1/<file>"}`. The retrieval was re-run on 2026-10-07 and
reproduced both digests exactly, which also reproduces the digests the literature mirror
recorded for `si/si9.xlsx` and `si/si4.xlsx`. Only these two files are deposited, because
they are the ones the loaders consumed for the first successful build; Supplementary Data
2 to 4 stay in the literature mirror and are quoted from there.

Supplementary Data 4 is deliberately not joined against: it lists the DBTL1 to DBTL6
CRISPRi arrays but covers cycles 1 and 3 to 6 only (360 rows, 287 distinct construct
names, no DBTL2), so the Source Data's own `cycle` column is the authority.

### Verification, L0 to L4

Run by `tests/torchcell/datasets/pputida/test_carruthers2025.py` (also runnable as a
script, which prints the table below). There is no `run_product_titer` runner in
`torchcell/verification/runners.py` yet; that belongs to plan step 6 and the exact
addition is named in the PR.

```
=== isoprenol titer ===
  [PASS] L0 structural: 465 records validated
  [PASS] L1 count: observed 465, expected 465
  [PASS] L2 value_fidelity: 465 titers finite and >= 0
  [PASS] L2 value_fidelity: 465 uncertainties finite and >= 0
  [PASS] L2 cross_method: 465 pairs agree within 1e-09 (titer_se == SD/sqrt(n))
  [PASS] L3 titer_unit_is_the_sources_mg_per_l_as_ug_per_ml
  [PASS] L3 every_genotype_carries_the_five_pathway_genes
  [PASS] L4 single_guide_titer_vs_supplementary_data_1: 120 overlapping entities agree within 0.005
=== proteome ===
  [PASS] L0 structural: 19 records validated
  [PASS] L1 count: observed 19, expected 19
  [PASS] L2 value_fidelity: 27056 values finite and >= 0
  [PASS] L2 cross_method: 19 pairs agree within 0.0 (1424 keys per record)
  [PASS] L3 every_protein_key_is_a_kt2440_locus_tag
  [PASS] L3 every_sample_is_a_biological_triplicate
  [PASS] L4 stored_target_profile_vs_released_sheet: 1424 overlapping entities agree within 1e-06
```

The titer L4 joins the built store to Supplementary Data 1, a DIFFERENT released file
from the one the loader reads. All 120 of its targets join a single-guide record, two more
than the construct-name join reaches, because `PP_1607` and `PP_4194` are released only
under the filler names `PP_1607_NT1` and `PP_4194_NT2` whose filler guide is not a
perturbation.

### Build

```bash
python -m torchcell.database.build_dataset_lmdb --dataset IsoprenolTiterCarruthers2025Dataset
python -m torchcell.database.build_dataset_lmdb --dataset ProteomeCarruthers2025Dataset
```

Both wrote `preprocess/build_manifest.json` with a 49-symbol closure, and both read
`fresh` with empty drift under `torchcell.provenance.build_manifest.check_all`. No KG
build was run and nothing under `$DATA_ROOT/database/` was touched.

### Open items for the owner

1. `ConcentrationUnit` has no `mg/L`. Adding it is additive but the enum is in every
   served closure, so it is a rebuild decision.
2. `ProductTiterExperiment.environment` and the other bacterial pairs should narrow to
   `CultureEnvironment`, or the fermentation vessel and volume cannot be stored.
3. `isoprenol` needs a `compound_identity_table.json` row, which means a line in
   `torchcell/datamodels/compound_identity_inputs/` and a human-run PubChem curation.
4. A `run_product_titer` / `run_bacterial_protein_abundance` verification runner, so the
   levels above run from `run_all` instead of from this dataset's test file.
5. The Dryad deposit needs a manual-browser retrieval before the full 472-strain proteome
   can be loaded.

## 2026.10.07 - Open item 4 closed: both families verify from run_all

The L0-L4 batteries printed above no longer live in
`tests/torchcell/datasets/pputida/test_carruthers2025.py`. They are
`carruthers2025.titer_report` and `carruthers2025.proteome_report`, reached through
`carruthers2025.verify_build(dataset_root, data_root, family=...)` the way the de Siqueira,
Kang, Lim and Caglar releases reach theirs, and
`torchcell.verification.runners.run_product_titer` /
`run_bacterial_protein_abundance` call them from `run_all`. Details, the full passing
output and the design reasons: [[torchcell.verification.runners]], 2026.10.07.

What changed in the levels themselves, all of it strictly additive:

- The titer L0-L3 now come from the new shared family verifier
  `torchcell.verification.product_titer.verify_product_titer_dataset`, so the unit decision,
  the `titer_se == SD/sqrt(n)` identity at 1e-9, the pathway-gene count and the product are
  checked the same way for every titer dataset. Two rules are new here: an uncertainty
  number and its type must both be stored or both be typed gaps, and the same for the
  replicate design.
- `strain_count_reconciles_with_the_papers_472` asserts the reconciliation this note
  documents rather than accepting either number: 465 strains plus the seven with six
  replicates is the paper's 472.
- The proteome family's `every_protein_key_is_a_kt2440_locus_tag` prefix check is replaced
  by the runner's `protein_and_perturbed_locus_containment_assembly`, which checks the 1,424
  keys and the perturbed host loci against the KT2440 locus universe read from the pinned
  assembly, and by `every_record_carries_the_same_protein_keys` for the per-record key set.
  The prefix check could not have caught a `PP_` tag the annotation does not carry.

**Item 2 is closed, and the statement above it is now stale.** The section "Environment"
says a `CultureEnvironment` placed in `ProductTiterExperiment.environment` dumps without its
culture-protocol slots, because the field was annotated `Environment`. It is annotated
`CultureEnvironment` now, and the built store carries the vessel: record 0's
`experiment.environment.culture_format` reads `vessel="48-well BioLector flower plate"`,
`working_volume_ul=1500.0`, `shaking_rpm=1000.0`, `endpoint="fixed_duration"`, with its own
`provenance` quote attached. Measured by reading the dev store read-only on 2026.10.07.
Items 1, 3 and 5 stand.

## 2026.10.08 - Four unstored titer panels and the overexpression proteome

Ranks 2 and 3 of [[plan.bacteria-si-phenotype-audit-pputida]]. Both dataset classes are
EXTENDED rather than joined by new ones, because the records are the same two experiment
classes and `GeneAdditionPerturbation` and `BacterialDeletionPerturbation` were already in
both datasets' 49-symbol schema closures (read off
`preprocess/build_manifest.json`). No adapter module, conf yaml,
`dataset_adapter_map` entry, `kg_bacteria.yaml` entry or adapter-case row changes, and the
`len(dataset_adapter_map) == 84` and `len(BACTERIAL) == 33` pins hold untouched. For the
served graph this is a superset admission, not a rebuild: the schema fingerprints are
unchanged and every stored record is still produced.

| family | before | added | after |
|---|---|---|---|
| `IsoprenolTiterCarruthers2025Dataset` | 465 | **37** | 502 |
| `ProteomeCarruthers2025Dataset` | 19 | **2** | 21 |

### What each panel contributes

| sheet | records | references | what one record is |
|---|---|---|---|
| `Figure 6a` (= `Supplementary Figure 11`) | 12 | 12 | a KO background carrying the sgRNA of the gene it deleted |
| `Figure 6d` | 4 | 0 new | a KO-only or CRISPRi-on-a-KO array strain |
| `Supplementary Figure 13d` | 19 | 1 (shared) | the ΔPP_0815 background carrying one off-target sgRNA |
| `Supplementary Figure 12bd` | 2 | 1 | an uninduced plasmid-borne native operon overexpression |
| `Supplementary Figure 12ac` | 2 | 2 | the same two strains' operon Top3 abundances |

The `Figure 6a` design is the Results' own, verbatim: "We sought to investigate the
presence of off-target gene downregulation amongst our best-performing sgRNAs by comparing
isoprenol titers between pairs of knockout (KO) strains harboring either a non-target sgRNA
or the "target" sgRNA previously used to downregulate the KO gene". So the Target arm is a
record and the Non-target arm is its `phenotype_reference`, since a non-targeting sgRNA
perturbs no gene. All 12 KO backgrounds are rows of Supplementary Table 2.

### The two internal duplications, resolved by measurement

**`Supplementary Figure 13d`'s reference IS `Figure 6a`'s, so it is stored once.** The
`Non-Target` triplicate (293.9713, 298.0681, 309.4453) is bit-identical to `Figure 6a`'s
`PP_0815` / `Non-target` triplicate. The build asserts the identity and builds ONE
reference object, shared by the `Figure 6a` ΔPP_0815 record and all 19
`Supplementary Figure 13d` records. The L3 row
`off_target_non_targeting_reference_is_stored_once` re-checks it on the built store: 20
records on that background, 1 distinct reference phenotype.

**The two `Target` triplicates are NOT the same cultures, and neither is stored as the
other.** `Figure 6a` gives (422.4448, 464.3335, 495.3173) and
`Supplementary Figure 13d` gives (450.4693, 464.3335, 477.8291). Measured across all 31
sheets of the workbook: `464.3335` occurs in both sheets, and each of the other four
occurs in exactly one sheet and nowhere else. So five distinct cultures exist and the two
figures plot overlapping draws. Both captions state `n = 3`
(Fig. 6a: "...demonstrating significant off-target effects of the PP_0815 sgRNA (n = 3)";
Supplementary Fig. 13: "All strains were cultured in triplicate (n = 3)"), so

- neither triplicate is stored as the other,
- neither is dropped, which would discard four released cultures, and
- they are NOT pooled into an `n = 5` design no caption states.

Each sheet's released triplicate is its own record and carries the disagreement in its
`preprocess/source_data_panels.csv` note. The cost is stated plainly: the one shared
culture is counted twice. `Supplementary Figure 13d` is the titer arm of the panel whose
proteome is already stored, and its 20 groups pair with `Supplementary Figure 13abc`'s 20
samples, so its triplicate is the proteomed batch's; `Figure 6a`'s is the KO-comparison
panel's. That is the reading, and it does not reach the level of settling which three
cultures are "the" triplicate, which is why neither is preferred.

**The CRISPRi arms are re-exports and none is taken.** Measured, and asserted in both
directions at build time: 33 of 33 `Figure 6a` and 99 of 99 `Figure 6d` `CRISPRi` cultures
are already-stored `Figure 4b` values, and 0 of the 190 cultures of the four new arms are.
A CRISPRi culture that stopped matching means the sheets diverged; a KO culture that
started matching means a titer is about to be stored twice. Both stop the build.

**`Figure 6d`'s control triplicate is three DBTL6 control cultures.** 157.9641, 157.7208
and 156.8139 are `Figure 4b`'s `Control_P1-R1`, `Control_P3-R2` and `Control_P2-R1`, all
cycle 6. So those four records reuse the stored DBTL6 `phenotype_reference` (n = 12)
instead of storing three of its twelve members a second time.

### `PP_0812-15` and the sgRNA the sheet does not name

`PP_0812-15` is an inclusive locus-number range: "PP_0815 (subunits of a terminal oxidase
complex PP_0812-15)", corroborated by Supplementary Table 5 building `IY1449b ΔPP_0812-15`
as `PP_0813-15` deleted from `IY1449b ΔPP_0812`. Its Target arm's sgRNA is named in exactly
one place in the mirror, the Results: "While most KO strains showed similar titers to their
CRISPRi counterparts from DBTL0 (Supplementary Fig. 11), ΔPP_0815 and ΔPP_0812-15 harboring
PP_0815 sgRNA produced significantly more isoprenol compared to those with non-targeting
guides, indicating that an off-target gene was driving isoprenol production level
(Fig. 6a)." No plasmid of Supplementary Data 3 carries a `PP_0812-15` spacer and
Supplementary Data 2's only `PP_0812-15` entry is the Cpf1 `PP_0812-15_Repair` oligo, so
`KO_MULTI_GENE_GUIDE` records that sentence and a NEW multi-gene background absent from the
map stops the build rather than getting a guessed guide.

### The overexpressed operons: a transposition in the sheet, a swap in the plasmid table

The Source Data labels its second overexpression strain `pSTABL2 (PP_2971-74)`. The loci
are **PP_2791-PP_2794**, stated by five mirrored sources: the Methods ("These operons,
PP_2208- PP_2209 and PP_2791-PP_2794, were amplified along with the DBTL3 Control
Vector..."), the Results ("we overexpressed two operons, PP_2791-94 (lvaA-D; levulinic acid
degradation) and PP_2208-09 (phnW-X; phosphonoacetalaldehyde hydrolase) on secondary
plasmids"), the Supplementary Fig. 12 caption's panels c and d (both "PP_2791-94"), the
Supplementary Fig. 10 caption ("PP_2793-94 encodes proteins in the levulinate degradation
pathway"), and Supplementary Data 3's own plasmid composition. The build PROVES it from the
samples' own proteome: `assert_overexpression_operons` requires the loci a label's samples
quantify to equal the operon, and records that the label names PP_2971-PP_2974 instead.

**The two plasmid NAMES are swapped between two pinned files.** Supplementary Data 3 says
`pSTABL1` is `pRSF1010-Gm-NagR-PP_2791-94` and `pSTABL2` is `pRSF1010-Gm-NagR-PP_2208-09`;
the Source Data labels its `pSTABL1_*` samples `pSTABL1 (PP_2208-09)` and measures PP_2208
and PP_2209 in them. Both statements are kept as `provenance` and NO plasmid name is stored
on a perturbation: `GeneAdditionPerturbation.construct_name` is `None`, and what the record
carries is the operon the sample's own measured proteome names.

The extra copies are `GeneAdditionPerturbation` with `is_heterologous=False` and
`source_organism="Pseudomonas putida"`, so the runner's `host_perturbed_gene_set` keeps
them and the containment gate checks the locus. NOT
`HeterologousPathwayPerturbation`, although that is the class carrying `gene_namespace`:
these operons are not part of the isoprenol pathway, and typing them there would make the
family's `heterologous_pathway_gene_counts` rule accept 7 or 9 where it accepts 5, which is
the rule that catches a production strain that lost its pathway.

### `Phnw` / `Phnx`: the near-miss that would have swapped two measurements

`Supplementary Figure 12ac` is the only sheet of the release that puts a locus tag in
`Protein.Description`, and for its two Phn rows it puts the WRONG one.

| statement | source | says |
|---|---|---|
| `PP_2208` is `phnX`, CDS product "phosphonoacetaldehyde hydrolase" | `GCA_000007565.2` (AAN67821.1) | PP_2208 is the hydrolase |
| `PP_2209` is `phnW`, CDS product "2-aminoethylphosphonate--pyruvate transaminase" | `GCA_000007565.2` (AAN67822.1) | PP_2209 is the transaminase |
| `Phnw` = Q88KT0 = `PHNW_PSEPK` = "2-aminoethylphosphonate--pyruvate transaminase" | Source Data `Supplementary Figure 13abc` | Q88KT0 is the transaminase |
| `Phnx` = Q88KT1 = `PHNX_PSEPK` = "Phosphonoacetaldehyde hydrolase" | same | Q88KT1 is the hydrolase |
| `Phnw` -> "PP_2208", `Phnx` -> "PP_2209" | Source Data `Supplementary Figure 12ac` | the opposite |

Matching on the FUNCTION both files state puts Q88KT0 at `PP_2209` and Q88KT1 at
`PP_2208`, which is what `reconcile_locus_tags`'s symbol layer resolves and what the 19
already-stored records are keyed by. The lone outlier is that one pair of description
cells. `assert_phn_crosswalk` asserts BOTH halves, so a corrected annotation or a corrected
sheet stops the build and the decision is re-made by hand rather than silently swapping two
measurements of one operon.

### Two things measured and NOT loaded, with the reason

**The six induced levels of `Supplementary Figure 12` (12 titer groups, 12 proteome
samples).** The inducer is salicylic acid ("The vector amplicons, designed for salicylic
acid induction of the inserted genes...") and its concentrations are released as bare
numbers (0, 31.25, 62.5, 125, 250, 500, 1000) in an `Inducer concentration` column with no
unit. Measured over the mirror: `salicyl` occurs once in `paper.md` (that sentence, with no
dose), `inducer` once in `si/si1.md` (the caption's "under various inducer concentrations",
with no unit), and the dose series appears nowhere in `paper.md`, `si/si1.md`, `si/si2.md`,
`si/si3.md` or `si/si8.md`. The deferral was followed: Supplementary Data 3 refers the
NagR-pNagAa vector to Supplementary Reference 1, "Yunus, I. S. et al. Predictive
genome-wide CRISPR-mediated gene downregulation for enhanced bioproduction", mirrored as
`yunusPredictiveCRISPRmediatedGene2026`, whose Methods state an L-arabinose dose and no
salicylate dose. `ConcentrationUnit` admits no unitless member and a `Concentration` needs
a value WITH a unit, so the six levels cannot be typed; collapsing them to one
`DoseBasis.fixed` would assert that one condition yielded six different titers. At level 0
no inducer was added, so the uninduced arm needs no unit and is loaded. It is also the arm
the paper's own claim rests on: "Uninduced expression, however, showed a ~10% increase in
titer compared to the RFP control."

**`Supplementary Figure 15` (7 records, 450 values).** Its basis is the percent-of-total
Top3 column, established by elimination over the three columns
`Supplementary Figure 12ac` releases from the same pipeline: all 540 of its cells lie in
0.00037341..3.43366 with zero negatives, while `Top_3pep_counts_mean` runs 0..1.0027e8 and
`log10_%_abundance` runs -2.5027..1.1373, leaving
`%_of protein_abundance_Top3-method` (0.0031..13.7169) as the only basis whose range
contains it. The caption's three rank claims also hold on that reading: column means
PMD\* 2.480 highest, mvaE 1.214 second, mvaS 0.589 lowest of the five pathway proteins,
dCas9 0.034 ("Pathway proteins typically followed a similar rank-order trend with PMD\*
being the highest expressed followed by mvaE. MvaS was typically the lowest expression
protein in the pathway. Owing to its toxicity, expression of dCas9 was kept comparatively
low.").

That is a DIFFERENT `measurement_type` from the raw Top3 signal every record of this family
stores, and `verify_protein_dataset`'s `measurement_type_consistent` rule exists to stop
exactly that mixing. Serving it therefore needs its own dataset class and the full adapter
gate. Two further blockers: five of its six keys are the pIY670 pathway tokens, which are
not loci of the pinned assembly and so fail the runner's
`protein_and_perturbed_locus_containment_assembly`, and the sixth names the dCas9 effector,
which this release carries on `CrisprConstruct` rather than as a gene with a systematic
name (the `Cas9` key is one of the 76 already dropped for that reason). Nothing it holds is
lost silently: its 90 cultures are the per-cycle controls whose titers are already this
dataset's `phenotype_reference`, with the counts 18, 12, 12, 12, 12, 12, 12 measured and
matching `CONTROL_N`.

Under the loadable-now arithmetic of the audit the two panels were counted at 49 + 21 = 70.
The accounting closes at 39: 12 induced titer groups, 12 induced proteome samples and the 7
`Supplementary Figure 15` records are the 31 difference, each with the reason above.

### Sourced statistics for the new records

- **`n_samples` = 3, uncertainty = sample SD**, for all four titer panels. Fig. 6
  caption: "a Comparison of mean isoprenol production between strains expressing either a
  PP_0815-targeting sgRNA or a non-targeting control sgRNA in a knockout background,
  demonstrating significant off-target effects of the PP_0815 sgRNA $\left( n = 3 \right)$
  ." Supplementary Figs. 11, 12 and 13 each restate "All strains were cultured in
  triplicate $( \mathsf { n } = 3 )$ and error bars represent standard deviation." The
  loader stores the count each group actually has: 3 for every group but the
  overexpression `Control` (6 cultures, two uninduced blocks under one name) and
  `Supplementary Figure 13d`'s `PP_0378` and `PP_3416` (2 cultures each).
- **The KO backgrounds** are "Stable gene knockouts were generated from the parent strain
  IY1449b via a Cpf1-mediated repair63." Each is a row of "Supplementary Table 2: List of
  Pseudomonas putida strains constructed in this study". The JBEI part ids stay in
  `preprocess/`, not on the perturbation, so a deletion object built here is bit-identical
  to the `PP_0815` deletion the proteome family already stores.
- **The overexpression strains** are "Plasmid sequences were verified by whole plasmid
  sequencing (Primordium Labs) and ultimately transformed into IY1449b with pIY670 to
  evaluate the impact of titrated induction on isoprenol titer", so the five pathway
  perturbations stand and the extra operon copy is added on top.
- **The RFP control** is "Strains harboring genes informed by Stabl were adapted to M9
  medium and cultured for over $4 8 \mathrm { h }$ in a Biolector Pro with an RFP control
  (JBx_266188) before GC-FID analysis." Supplementary Data 3 row `pTE519` / `JBx_266188` is
  `pRSF1010-Gm-mCherry`. It is a `phenotype_reference`, which carries no genotype, so the
  mCherry marker is recorded here rather than typed.
- **The overexpression endpoint is a typed gap.** That same sentence states the endpoint as
  "over 48 h" while the shared culturing Methods state "Following $4 8 \mathrm { h }$ of
  production", so those four records carry `duration_hours=None` with a
  `ProvenanceGap(field="duration_hours")` rather than the shared 48.0. Everything else the
  sentence leaves to the shared protocol and does not contradict (M9-NREL, 24 C, the flower
  plate, the L-arabinose induction) is carried unchanged.
- **The `Supplementary Figure 13d` panel size** is Supplementary Table 1's: 14 off-target
  gene targets, 18 released samples with the repeated cultures, which is why
  "d) Isoprenol titers of off-target sgRNA strains failed to recapitulate the observed
  titer of ΔPP_0815 harboring the PP_0815 sgRNA." gives 19 records plus the one reference.

### Verification, L0 to L4, on the rebuilt dev stores

```
isoprenol_titer_carruthers2025: PASS
  [ok] L0 structural: 502 records validated
  [ok] L1 count: observed 502, expected 502
  [ok] L1 campaign_and_panel_records_partition: 465 Figure 4b campaign strains + 37 Source Data panel strains = 502
  [ok] L1 strain_count_reconciles_with_the_papers_472: 465 campaign strains + 7 with six replicates = 472 (the paper's 472)
  [ok] L2 value_fidelity: 502 values checked
  [ok] L2 uncertainty_nonnegative: 502 values checked
  [ok] L2 se_is_the_uncertainty_over_sqrt_n: 502 pairs agree within 1e-09
  [ok] L3 titer_unit_is_the_pinned_unit: stored units ['ug/mL']
  [ok] L3 uncertainty_is_typed_or_gapped
  [ok] L3 replicate_design_is_sourced_or_gapped
  [ok] L3 heterologous_pathway_gene_counts: per-record counts [5]; the dataset declares [5]
  [ok] L3 product_is_the_declared_one: stored products ['isoprenol']
  [ok] L3 off_target_non_targeting_reference_is_stored_once: 20 records on the ΔPP_0815 background share 1 distinct non-targeting reference phenotype
  [ok] L4 single_guide_titer_vs_supplementary_data_1: 120 overlapping entities agree within 0.005
  [ok] L4 ko_array_titer_vs_results_text: 2 overlapping entities agree within 1.0
  [ok] L4 panel_titers_vs_released_sheets: 37 overlapping entities agree within 1e-09
  [ok] L4 perturbed_gene_containment_assembly: 1.000 of 141 measured genes are loci of pputida_KT2440_ASM756v2
proteome_carruthers2025: PASS
  [ok] L0 structural: 21 records validated
  [ok] L1 count: observed 21, expected 21
  [ok] L1 orf_uniqueness: 26 ORFs, 8 with multiple strains (expected)
  [ok] L1 protein_key_set_sizes_are_the_panels: 21 records over key-set sizes {2: 1, 4: 1, 1424: 19}
  [ok] L2 value_fidelity: 27062 values checked
  [ok] L2 se_nonnegative: 27062 values checked
  [ok] L3 reference_finite: finite + key-matched for all 27062 values
  [ok] L3 measurement_type_consistent: single measurement_type: 'dia_nn_top3_peptide_signal_mean'
  [ok] L3 every_sample_is_a_biological_triplicate: stored replicate counts [3]
  [ok] L4 stored_target_profile_vs_released_sheet: 1424 overlapping entities agree within 1e-06
  [ok] L4 overexpression_proteome_vs_released_sheet: 6 overlapping entities agree within 1e-06
  [ok] L4 protein_and_perturbed_locus_containment_assembly: 1.000 of 1425 measured genes are loci of pputida_KT2440_ASM756v2
```

Three new rules are this revision's own. `campaign_and_panel_records_partition` asserts the
stored half of the partition the build proves on the released bytes.
`ko_array_titer_vs_results_text` joins the prose: "Combining KOs with specific sgRNAs for
PP_0528 and PP_0815 further improved titer to 4-fold that of the control $( 6 5 1 \mathrm {
m g / L } )$ and $12 \%$ more isoprenol than the two-sgRNA array in a strain without KOs $(
5 8 0 \mathrm { m g / L }$ , $p < 0 . 0 2 )$" names one NEW `Figure 6d` record and one
already-stored `Figure 4b` array, so one sentence joins the panel and the campaign at once.
Stored 651.860 against 651 and 579.534 against 580; the text prints both to integer mg/L
and is not consistent about rounding, so the tolerance is that unit.
`panel_titers_vs_released_sheets` re-reads all four panels and re-derives every stored
mean, in both directions, so a stored panel titer matching no released group fails too.

Two existing rules are now SCOPED to the campaign records, by a predicate that asks whether
a record carries a deletion or an extra native copy. `strain_count_reconciles_with_the_papers_472`
must not count KO and overexpression strains the abstract's 472 does not, and
`single_guide_titer_vs_supplementary_data_1` must not join a ΔPP_0815 strain carrying the
PP_0815 sgRNA to a DBTL0 single-guide mean: it has exactly one CRISPRi perturbation on the
same gene, so the nearest-record rule would have hidden a false join.
`every_record_carries_the_same_protein_keys` is replaced by
`protein_key_set_sizes_are_the_panels`, because `Supplementary Figure 12ac` quantifies only
the operon a sample overexpresses: the pinned multiset is 19 x 1,424 plus one 2 and one 4.

### New preprocess artifacts

- `isoprenol_titer_carruthers2025/preprocess/source_data_panels.csv` -- the 37 panel
  records with their deletions, knockdowns, native copies, reference key and note.
- `isoprenol_titer_carruthers2025/preprocess/panel_proofs.json` -- the 12 measured
  cross-sheet statements the build asserted.
- `proteome_carruthers2025/preprocess/panel_proofs.json` -- the 5 statements of the
  crosswalk, the operon correction and the uninduced pairing.
- `proteome_carruthers2025/preprocess/samples.csv` gains a `sheet` column.

### Open items for the owner, added to the five above

6. `GeneAdditionPerturbation` declares no `gene_namespace` (the field lives on
   `HeterologousPathwayPerturbation`), so a bacterial extra NATIVE copy cannot state the
   namespace of its locus tag on the leaf; it is recoverable from the record's
   `genome_reference`. Adding the field is additive but touches a class in every served
   closure.
7. `ConcentrationUnit` has no way to carry a released dose whose unit the source never
   states. Twenty-four of this release's records wait on it, and the honest alternatives
   (a unitless member, or a relative-level field on the perturbation) are both schema
   decisions, not loader ones.
8. Serving `Supplementary Figure 15` needs its own `ProteinAbundance` dataset class for the
   percent-of-total basis, plus a containment rule that admits a protein key the record's
   own genotype declares heterologous. The mirror image of the de Siqueira finding (rank 5
   of the audit), so one decision unblocks both.
9. **Rank 8 of the audit, the knockdown ratios, is still CONDITIONAL and nothing here
   changes that.** This branch did not touch it. Working view, unverified by any code
   written here: the contradiction is real, since `ProteinAbundancePhenotype`'s docstring
   forbids a ratio while the Yunus loader stores one, and `Figure 3c` must be the authority
   over Supplementary Data 1 because the audit measured Supplementary Data 1's two
   expression headers to be swapped. The ratio question is the same one as item 8's: both
   want a protein phenotype whose value is not an absolute.
