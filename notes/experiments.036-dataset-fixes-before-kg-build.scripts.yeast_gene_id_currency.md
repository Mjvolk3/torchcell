---
id: ts1p994d7tgs8oqr9bwol2p
title: Yeast_gene_id_currency
desc: ''
updated: 1791701433265
created: 1791701433265
---

## 2026.10.11 - Every released ORF R64-4-1 no longer carries, classified (#893, #930, #979, #897 to #900)

Run under slurm job 3829 (`gh_yeast_gene_id_currency.slurm`, 5 min 23 s, MaxRSS 7.2 GB) at commit `a41c2628b`; results in `experiments/036-dataset-fixes-before-kg-build/results/yeast_gene_id_currency.json`, tables below rendered by `--table`.

Measured: 51 distinct released ORFs are not live R64-4-1 loci, 82 store occurrences (35 Costanzo 2016 + 47 Kuzmin 2020): 16 renamed, 28 merged, 7 retired. The set of names the release carries equals the set [[torchcell.datasets.systematic_name_currency]] decides (`tables_cover_exactly_the_measured_names: true`), and the genome agrees with every decision.

The release's own strains support each class. Every merged ORF sits beside a separate deletion strain of its current locus with a different strain number (`YAR044W_dma74` beside `YAR042W_dma76`), so mapping it would put two different deletions on one gene. For 13 of the 16 renamed ORFs, Costanzo 2016 names the same strain number by the current ORF (`YAL043C-A_dma42` in Kuzmin 2020 is `YAL042C-A_dma42` in Costanzo 2016). YFL035C-B is renamed but its locus YFL034C-A also has its own strain (`YFL034C-A_dma1493` beside `YFL035C-B_dma1491`); after the rename both are stored as YFL034C-A, distinguished by strain id. Ten of the 16 renamed loci are Dubious ORFs that R64-4-1 still carries.

Per-store counts are what each loader's currency check does to its input frame; "imp2' rows" are Kuzmin 2020 rows whose stored common name gains the prime (YIL154C).

| Released ORF | Releases | R64-4-1 | Current locus (type, class) | Decision | Strains of the ORF / of its locus |
|---|---|---|---|---|---|
| YAL043C-A | kuzmin2020 | renamed | YAL042C-A (gene, Dubious) | renamed | costanzo2016: - / YAL042C-A_dma42; kuzmin2020: YAL043C-A_dma42 / - |
| YAL058C-A | kuzmin2020 | renamed | YAL056C-A (gene, Dubious) | renamed | costanzo2016: - / YAL056C-A_dma57; kuzmin2020: YAL058C-A_dma57 / - |
| YAR037W | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YAR037W_dma78 / -; kuzmin2020: YAR037W_dma78 / - |
| YAR040C | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YAR040C_dma77 / -; kuzmin2020: YAR040C_dma77 / - |
| YAR043C | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YAR043C_dma75 / -; kuzmin2020: YAR043C_dma75 / - |
| YAR044W | costanzo2016, kuzmin2020 | renamed | YAR042W (gene, Verified) | merged | costanzo2016: YAR044W_dma74 / YAR042W_dma76, YAR042W_sn3004; kuzmin2020: YAR044W_dma74 / YAR042W_dma76 |
| YBR075W | costanzo2016, kuzmin2020 | renamed | YBR074W (gene, Verified) | merged | costanzo2016: YBR075W_dma248 / YBR074W_dma249; kuzmin2020: YBR075W_dma248 / YBR074W_dma249 |
| YBR090C-A | kuzmin2020 | renamed | YBR089C-A (gene, Verified) | renamed | costanzo2016: - / YBR089C-A_sn1997; kuzmin2020: YBR090C-A_dma263 / - |
| YBR100W | costanzo2016, kuzmin2020 | renamed | YBR098W (gene, Verified) | merged | costanzo2016: YBR100W_dma256 / YBR098W_dma258, YBR098W_sn1294; kuzmin2020: YBR100W_dma256 / YBR098W_dma258 |
| YCL006C | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YCL006C_dma457 / -; kuzmin2020: YCL006C_dma457 / - |
| YCL012W | costanzo2016, kuzmin2020 | renamed | YCL014W (gene, Verified) | merged | costanzo2016: YCL012W_dma476 / YCL014W_dma474, YCL014W_sn1239; kuzmin2020: YCL012W_dma476 / YCL014W_dma474 |
| YCL013W | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YCL013W_dma475 / -; kuzmin2020: YCL013W_dma475 / - |
| YCL026C | costanzo2016, kuzmin2020 | retired | - | retired | costanzo2016: YCL026C_dma504 / -; kuzmin2020: YCL026C_dma504 / - |
| YCL060C | costanzo2016, kuzmin2020 | renamed | YCL061C (gene, Verified) | merged | costanzo2016: YCL060C_dma523 / YCL061C_dma522, YCL061C_sn1212; kuzmin2020: YCL060C_dma523 / YCL061C_dma522 |
| YCL062W | costanzo2016, kuzmin2020 | renamed | YCL063W (gene, Verified) | merged | costanzo2016: YCL062W_dma521 / YCL063W_dma520, YCL063W_sn4689; kuzmin2020: YCL062W_dma521 / YCL063W_dma520 |
| YCR062W | costanzo2016, kuzmin2020 | renamed | YCR061W (gene, Verified) | merged | costanzo2016: YCR062W_dma587 / YCR061W_dma566, YCR061W_sn3432; kuzmin2020: YCR062W_dma587 / YCR061W_dma566 |
| YDL038C | costanzo2016, kuzmin2020 | renamed | YDL039C (gene, Verified) | merged | costanzo2016: YDL038C_dma625, YDL038C_sn3130 / YDL039C_dma647, YDL039C_sn797; kuzmin2020: YDL038C_dma625 / YDL039C_dma647 |
| YDR474C | costanzo2016, kuzmin2020 | renamed | YDR475C (gene, Verified) | merged | costanzo2016: YDR474C_dma1193 / YDR475C_dma1192, YDR475C_sn3071; kuzmin2020: YDR474C_dma1193 / YDR475C, YDR475C_dma1192 |
| YER108C | costanzo2016, kuzmin2020 | non_gene_feature | YER109C (blocked_reading_frame, None) | merged | costanzo2016: YER108C_dma1396 / YER109C_dma1395; kuzmin2020: YER108C_dma1396 / YER109C_dma1395 |
| YFL006W | costanzo2016, kuzmin2020 | renamed | YFL007W (gene, Verified) | merged | costanzo2016: YFL006W_dma1487 / YFL007W_dma5286, YFL007W_sn3847; kuzmin2020: YFL006W_dma1487 / YFL007W_dma5286 |
| YFL035C-B | kuzmin2020 | renamed | YFL034C-A (gene, Verified) | renamed | costanzo2016: - / YFL034C-A_dma1493; kuzmin2020: YFL035C-B_dma1491 / YFL034C-A_dma1493 |
| YFL043C | costanzo2016, kuzmin2020 | renamed | YFL042C (gene, Verified) | merged | costanzo2016: YFL043C_dma1509 / YFL042C_dma1510, YFL042C_sn3179; kuzmin2020: YFL043C_dma1509 / YFL042C, YFL042C_dma1510 |
| YFL057C | costanzo2016 | non_gene_feature | YFL056C (pseudogene, None) | merged | costanzo2016: YFL057C_sn2744 / YFL056C_dma1520, YFL056C_sn2747; kuzmin2020: - / YFL056C_dma1520 |
| YFR024C | costanzo2016, kuzmin2020 | renamed | YFR024C-A (gene, Verified) | merged | costanzo2016: YFR024C_dma1570 / YFR024C-A_dma1569, YFR024C-A_sn561; kuzmin2020: YFR024C_dma1570 / YFR024C-A_dma1569 |
| YGL046W | costanzo2016, kuzmin2020 | renamed | YGL045W (gene, Verified) | merged | costanzo2016: YGL046W_dma1629 / YGL045W_dma1630, YGL045W_sn1386; kuzmin2020: YGL046W_dma1629 / YGL045W_dma1630 |
| YHR079C-B | kuzmin2020 | renamed | YHR079C-A (gene, Verified) | renamed | costanzo2016: - / YHR079C-A_dma2128, YHR079C-A_sn1427; kuzmin2020: YHR079C-B_dma2128 / - |
| YIL015C-A | kuzmin2020 | renamed | YIL014C-A (gene, Uncharacterized) | renamed | costanzo2016: - / YIL014C-A_dma2248, YIL014C-A_sn4712; kuzmin2020: YIL015C-A_dma2248 / - |
| YIL080W | costanzo2016 | retired | - | retired | costanzo2016: YIL080W_sn3882 / - |
| YIL168W | costanzo2016, kuzmin2020 | non_gene_feature | YIL167W (blocked_reading_frame, None) | merged | costanzo2016: YIL168W_dma2393, YIL168W_sn4403 / YIL167W_dma2394, YIL167W_sn4320; kuzmin2020: YIL168W_dma2393 / YIL167W_dma2394 |
| YIR020W-B | kuzmin2020 | renamed | YIR020W-A (gene, Dubious) | renamed | costanzo2016: - / YIR020W-A_dma2436; kuzmin2020: YIR020W-B_dma2436 / - |
| YIR044C | costanzo2016 | non_gene_feature | YIR043C (blocked_reading_frame, None) | merged | costanzo2016: YIR044C_sn3249 / YIR043C_dma2441; kuzmin2020: - / YIR043C_dma2441 |
| YJL017W | costanzo2016, kuzmin2020 | renamed | YJL016W (gene, Uncharacterized) | merged | costanzo2016: YJL017W_dma2457 / YJL016W_dma2458, YJL016W_sn3466; kuzmin2020: YJL017W_dma2457 / YJL016W_dma2458 |
| YJL021C | costanzo2016, kuzmin2020 | renamed | YJL020C (gene, Verified) | merged | costanzo2016: YJL021C_dma2455 / YJL020C_dma2456, YJL020C_sn552; kuzmin2020: YJL021C_dma2455 / YJL020C_dma2456 |
| YJL206C-A | kuzmin2020 | renamed | YJL205C (gene, Verified) | renamed | costanzo2016: - / YJL205C_dma2602, YJL205C_sn4717; kuzmin2020: YJL206C-A_dma2602 / - |
| YKL158W | costanzo2016, kuzmin2020 | renamed | YKL157W (gene, Verified) | merged | costanzo2016: YKL158W_dma2879 / YKL157W_dma2880, YKL157W_sn4419; kuzmin2020: YKL158W_dma2879 / YKL157W_dma2880, YKL157W_tm1219, YKL157W_tm1636 |
| YKL199C | costanzo2016, kuzmin2020 | renamed | YKL198C (gene, Verified) | merged | costanzo2016: YKL199C_dma2898 / YKL198C_dma2899; kuzmin2020: YKL199C_dma2898 / YKL198C_dma2899 |
| YKL200C | costanzo2016, kuzmin2020 | renamed | YKL201C (gene, Verified) | merged | costanzo2016: YKL200C_dma2897 / YKL201C_dma2896; kuzmin2020: YKL200C_dma2897 / YKL201C_dma2896 |
| YLR312C-B | costanzo2016 | renamed | YLR313C (gene, Verified) | merged | costanzo2016: YLR312C-B_sn4563 / YLR313C_dma3337, YLR313C_sn572; kuzmin2020: - / YLR313C_dma3337 |
| YLR391W | costanzo2016, kuzmin2020 | renamed | YLR390W-A (gene, Verified) | merged | costanzo2016: YLR391W_dma3428 / YLR390W-A_dma3429, YLR390W-A_sn736; kuzmin2020: YLR391W_dma3428 / YLR390W-A_dma3429 |
| YML010C-B | kuzmin2020 | renamed | YML009C-A (gene, Dubious) | renamed | costanzo2016: - / YML009C-A_dma3525; kuzmin2020: YML010C-B_dma3525 / - |
| YML033W | costanzo2016, kuzmin2020 | renamed | YML034W (gene, Verified) | merged | costanzo2016: YML033W_dma3530 / YML034W_dma3529, YML034W_sn1459; kuzmin2020: YML033W_dma3530 / YML034W_dma3529, YML034W_tm1513, YML034W_tm386 |
| YML035C-A | kuzmin2020 | renamed | YML034C-A (gene, Dubious) | renamed | costanzo2016: - / YML034C-A_dma3550; kuzmin2020: YML035C-A_dma3550 / - |
| YML048W-A | kuzmin2020 | renamed | YML047W-A (gene, Dubious) | renamed | costanzo2016: - / YML047W-A_dma3542; kuzmin2020: YML048W-A_dma3542 / - |
| YML058C-A | kuzmin2020 | renamed | YML057C-A (gene, Dubious) | renamed | costanzo2016: - / YML057C-A_dma3570; kuzmin2020: YML058C-A_dma3570 / - |
| YML102C-A | kuzmin2020 | renamed | YML101C-A (gene, Dubious) | renamed | kuzmin2020: YML102C-A_dma3591 / - |
| YML117W-A | kuzmin2020 | renamed | YML116W-A (gene, Dubious) | renamed | costanzo2016: - / YML116W-A_dma3603; kuzmin2020: YML117W-A_dma3603 / - |
| YMR158W-A | kuzmin2020 | renamed | YMR158W-B (gene, Dubious) | renamed | costanzo2016: - / YMR158W-B_dma3768; kuzmin2020: YMR158W-A_dma3768 / - |
| YOL053C-A | kuzmin2020 | renamed | YOL052C-A (gene, Verified) | renamed | costanzo2016: - / YOL052C-A_dma4337, YOL052C-A_sn4459; kuzmin2020: YOL053C-A_dma4337 / - |
| YOR088W | costanzo2016, kuzmin2020 | renamed | YOR087W (gene, Verified) | merged | costanzo2016: YOR088W_dma4521 / YOR087W_dma4522, YOR087W_sn2444; kuzmin2020: YOR088W_dma4521 / YOR087W_dma4522 |
| YOR240W | costanzo2016, kuzmin2020 | renamed | YOR239W (gene, Verified) | merged | costanzo2016: YOR240W_dma5230 / YOR239W_dma4652; kuzmin2020: YOR240W_dma5230 / YOR239W_dma4652 |
| YPR090W | costanzo2016, kuzmin2020 | renamed | YPR089W (gene, Uncharacterized) | merged | costanzo2016: YPR090W_dma5080 / YPR089W_dma5081, YPR089W_sn3399; kuzmin2020: YPR090W_dma5080 / YPR089W_dma5081 |

| Store | Rows in | Refused | Rows out | Renamed rows | imp2' rows | Names |
|---|---:|---:|---:|---:|---:|---:|
| smf_costanzo2016 | 20,484 | 74 | 20,410 | 0 | 0 | 35 |
| dmf_costanzo2016 | 20,705,612 | 157,605 | 20,548,007 | 0 | 0 | 35 |
| dmi_costanzo2016 | 20,705,612 | 157,605 | 20,548,007 | 0 | 0 | 35 |
| smf_kuzmin2020 | 472 | 0 | 472 | 0 | 0 | 0 |
| dmf_kuzmin2020 | 632,998 | 658 | 632,340 | 342 | 488 | 47 |
| tmf_kuzmin2020 | 301,798 | 317 | 301,481 | 166 | 238 | 47 |
| dmi_kuzmin2020 | 632,797 | 658 | 632,139 | 342 | 488 | 47 |
| tmi_kuzmin2020 | 301,798 | 317 | 301,481 | 166 | 238 | 47 |
