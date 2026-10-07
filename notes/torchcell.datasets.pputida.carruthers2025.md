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
