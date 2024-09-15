---
id: rdi64el69j2nxjlvtrcftqu
title: Kuzmin2020
desc: ''
updated: 1725990105586
created: 1723828982525
---

## LLM Prompt for Initial Build of Dataset Before Debug

We are going to use raw data from `kuzmin2020` to create the following dataset:

- `SmfKuzmin2020Dataset`
- `DmfKuzmin2020Dataset`
- `TmfKuzmin2020Dataset`
- `DmiKuzmin2020Dataset`
- `TmiKuzmin2020Dataset`

Here is the new link address. "<https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip>"

from within this zip we need aaz5667-Table-S1... it has 794774 total rows

Here is a snippets of the `xlsx` file with headers.

Table S1. Raw genetic interaction dataset. This file contains digenic interaction scores as well as raw and adjusted trigenic interaction scores with 12 columns.

```xlsx
Query strain ID Query allele name Array strain ID Array allele name Combined mutant type Raw genetic interaction score (epsilon) Adjusted genetic interaction score (epsilon or tau) P-value Query single/double mutant fitness Array single mutant fitness Double/triple mutant fitness Double/triple mutant fitness standard deviation
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL007C_dma91 sla1Δ digenic 0.007534 0.007534 4.50E-01  0.962 0.9695 0.0465
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL008W_dma90 hir1Δ digenic -0.008696 -0.008696 3.30E-01  1.02 1.0113 0.0123
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL021C_dma117 hap3Δ digenic -0.096643 -0.096643 1.68E-01  0.852 0.7554 0.1147
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL023C_tsa111 mcm2-1 digenic 0.005416 0.005416 4.65E-01  0.909 0.9144 0.0479
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL024W_dma116 ncl1Δ digenic -0.047343 -0.047343 1.81E-01  0.966 0.9187 0.0444
```

Notice that just like kuzmin 2018 we have to deal with YDL227C which is a hoΔ. This ids part of total number of gene interactions.

We can combine Table-S1 with S3 directly

Table S3. Raw genetic interaction dataset from pilot screens. This file contains digenic interaction scores as well as raw and adjusted trigenic interaction scores in a tab-delimited format with 12 columns.

This is file name aaz5667-Table-S3.xlsx

```
Query strain ID Query allele name Array strain ID Array allele name Combined mutant type Raw genetic interaction score (epsilon) Adjusted genetic interaction score (epsilon or tau) P-value Query single/double mutant fitness Array single mutant fitness Double/triple mutant fitness Double/triple mutant fitness standard deviation
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL002W_dma23 vps8Δ digenic 0.052845 0.052845 1.98E-01  0.764 0.8168 0.0501
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL004W_dma22 yal004wΔ digenic -0.00189 -0.00189 4.90E-01  1.009 1.0071 0.075
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL005C_dma21 ssa1Δ digenic 0.006864 0.006864 4.61E-01  1.0405 1.0474 0.0761
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL007C_dma20 erp2Δ digenic -0.020312 -0.020312 3.80E-01  1.026 1.0057 0.0699
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL008W_dma19 fun14Δ digenic 0.002584 0.002584 4.82E-01  1.0445 1.0471 0.0497
...
```

We also need to use Table S5 aaz5667-Table-S5.xlsx. Their is either 'Double mutant' or 'Single mutant' here

```xlsx
"Table S5. Fitness standard for single and double mutant query strains and query-query interactions.
This file contains the fitness standard for single and double mutant query strains as well as the genetic interaction between the query genes. "         
Query Strain ID ORF1 ORF2 Gene1 Gene2 Allele1 Allele2 Mutant type Fitness St.dev.
tm1 YAR002C-A YGL002W ERP1 ERP6 erp1Δ erp6Δ Double mutant 0.87051 0.007
tm10 YDR492W YOL101C IZH1 IZH4 izh1Δ izh4Δ Double mutant 0.89413 0.005
tm100 YIR037W YKL026C HYR1 GPX1 hyr1Δ gpx1Δ Double mutant 0.99379 0.0045
tm101 YDR146C YLR131C SWI5 ACE2 swi5Δ ace2Δ Double mutant NaN NaN
tm103 YIL117C YNL058C PRM5 YNL058C prm5Δ ynl058cΔ Double mutant 0.92621 0.0055
tm104 YMR182C YPL230W RGM1 USV1 rgm1Δ usv1Δ Double mutant 0.88059 0.0045
...
```

To summarize the files that will be necessary for each dataset. These should also be the files that are used to create the raw dataset.

- `SmfKuzmin2020Dataset`
  files needed:
  - aaz5667-Table-S5.xlsx
- `DmfKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
  - aaz5667-Table-S3.xlsx
  - aaz5667-Table-S5.xlsx
- `TmfKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
- `DmiKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
- `TmiKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx

To implement this make sure we follow the patterns laid out by `kuzmin2018` datasets, and adhere to the schema.py.

This is the DOI: 10.1126/science.aaz5667, and pubmed id 32586993 and pubmed url <https://pubmed.ncbi.nlm.nih.gov/32586993/>

## 2024.09.10 - There are Negative Fitness Values

I didn't think we would ever encounter this since the fitness score is computed as the ratio of mutant over wild type. This is likely an artifact of data processing, normalization, standardization, etc. Based on this definition it is strange to keep the negative as it violates our current data validation. [[torchcell.datamodels.schema]]. I think for this instance we should correct to 0.0 with the validator but provide a warning. I think this is okay given so few data points are negative and for all practical purpose negative means `0.0` or a dead cell.

Table S3.

| Query strain ID        | Query allele name | Array strain ID | Array allele name | Combined mutant type | Raw genetic interaction score (epsilon) | Adjusted genetic interaction score (epsilon or tau) | P-value  | Query single/double mutant fitness | Array single mutant fitness | Double/triple mutant fitness | Double/triple mutant fitness standard deviation |
| ---------------------- | ----------------- | --------------- | ----------------- | -------------------- | --------------------------------------- | --------------------------------------------------- | -------- | ---------------------------------- | --------------------------- | ---------------------------- | ----------------------------------------------- |
| YMR198W+YDL227C_tm467  | cik1Δ+hoΔ         | YPR162C_tsa1247 | orc4-5001         | digenic              | -0.718108                               | -0.718108                                           | 1.71E-57 |                                    | 0.6379                      | -0.0802                      | 0.0671                                          |
| YAL056W+YOR371C_tm1601 | gpb2Δ+gpb1Δ       | YNL102W_tsa345  | pol1-17           | trigenic             | -0.523199                               | 0.160684                                            | 0.00E+00 | 0.9083                             | 0.5481                      | -0.0254                      | 0                                               |
| YKR095W+YIL149C_tm1623 | mlp1Δ+mlp2Δ       | YMR197C_tsa635  | vti1-11-supp1     | trigenic             | -0.630539                               | -0.818542                                           | 4.35E-35 | 0.8307                             | 0.7494                      | -0.008                       | 0.0559                                          |

Table S1.

| Query strain ID        | Query allele name | Array strain ID | Array allele name | Combined mutant type | Raw genetic interaction score (epsilon) | Adjusted genetic interaction score (epsilon or tau) | P-value   | Query single/double mutant fitness | Array single mutant fitness | Double/triple mutant fitness | Double/triple mutant fitness standard deviation |
| ---------------------- | ----------------- | --------------- | ----------------- | -------------------- | --------------------------------------- | --------------------------------------------------- | --------- | ---------------------------------- | --------------------------- | ---------------------------- | ----------------------------------------------- |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YBR131W_dma294  | ccz1Δ             | digenic              | -0.872019                               | -0.872019                                           | 0.00E+00  |                                    | 0.744                       | -0.128                       | 0                                               |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YPL226W_dma4968 | new1Δ             | digenic              | -0.375281                               | -0.375281                                           | 1.11E-85  |                                    | 0.327                       | -0.0483                      | 0.0149                                          |
| YHR030C+YKL161C_tm61   | slt2Δ+kdx1Δ       | YGR157W_dma1899 | cho2Δ             | trigenic             | -0.256726                               | -0.045604                                           | 8.74E-38  | 0.8665                             | 0.247                       | -0.0427                      | 0.0196                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YFL007W_dma5286 | blm10Δ            | digenic              | -1.077915                               | -1.077915                                           | 0.00E+00  |                                    | 1.036                       | -0.0419                      | 0.0275                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YHR003C_dma2083 | tcd1Δ             | digenic              | -1.006909                               | -1.006909                                           | 0.00E+00  |                                    | 0.965                       | -0.0419                      | 0                                               |
| YCR069W+YNR028W_tm163  | cpr4Δ+cpr8Δ       | YGL023C_dma1597 | pib2Δ             | trigenic             | -0.876595                               | -0.850199                                           | 2.52E-61  | 0.8946                             | 0.943                       | -0.033                       | 0.056                                           |
| YHR030C+YDL227C_tm432  | slt2Δ+hoΔ         | YGR157W_dma1899 | cho2Δ             | digenic              | -0.280035                               | -0.280035                                           | 1.89E-12  |                                    | 0.247                       | -0.033                       | 0.0527                                          |
| YDR077W+YER150W_tm3410 | sed1Δ+spi1Δ       | YAL025C_tsa1066 | MAK16-ph          | trigenic             | -0.741312                               | -0.722245                                           | 0.00E+00  |                                    | 0.716                       | -0.0253                      | 0                                               |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YKR092C_dma3002 | srp40Δ            | digenic              | -0.952263                               | -0.952263                                           | 0.00E+00  |                                    | 0.927                       | -0.0253                      | 0                                               |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YLR077W_dma3164 | fmp25Δ            | digenic              | -1.012256                               | -1.012256                                           | 0.00E+00  |                                    | 0.996                       | -0.0163                      | 0.0119                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YLR077W_dma3164 | fmp25Δ            | digenic              | -1.011758                               | -1.011758                                           | 4.23E-120 |                                    | 0.996                       | -0.0158                      | 0.048                                           |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YPL226W_dma4968 | new1Δ             | digenic              | -0.342596                               | -0.342596                                           | 0.00E+00  |                                    | 0.327                       | -0.0156                      | 0                                               |
| YMR199W+YPL256C_tm1604 | cln1Δ+cln2Δ       | YLR337C_dma3380 | vrp1Δ             | trigenic             | -0.355614                               | -0.355614                                           | 5.81E-03  |                                    | 0.53                        | -0.0156                      | 0.0901                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YGR121C_dma1881 | mep1Δ             | digenic              | -1.024336                               | -1.024336                                           | 0.00E+00  |                                    | 1.01                        | -0.0143                      | 0.0209                                          |
| YAL056W+YDL227C_tm1888 | gpb2Δ+hoΔ         | YHR081W_dma2126 | lrp1Δ             | digenic              | -0.499073                               | -0.499073                                           | 3.99E-09  |                                    | 0.49                        | -0.0091                      | 0.2276                                          |
