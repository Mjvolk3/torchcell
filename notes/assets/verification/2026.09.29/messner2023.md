# Messner 2023 KO proteome (`ProteomeMessner2023Dataset`) re-verification, 2026-09-29

> Provenance: raw report of a read-only Fable 5.1 verifier agent, 2026-09-29, copied unchanged from the session scratchpad `messner2023.md` (sha256 `6e78fab0921d901ac053d2a7af675b1c20674ad0609ba4b05669ca6c51009656`); the scratch scripts it cites were not committed. Consolidated in [[datasets.showcase-verification.2026.09.29]].

Read-only verification. No file under the repo or `$DATA_ROOT` was touched; scratch scripts and
downloaded web records sit beside this report in
`/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/813cd5c0-bf08-48e1-91f0-98bae9709442/scratchpad/verify/`
(`messner_rawfiles.py`, `messner_lmdb.py`, `messner_batch.py`, `mendeley_files.json`,
`mendeley_record.json`, `summary_fileupload.pdf/.txt`, `px_PXD036062.xml`, `massive_proxi.json`,
`massive_files.html`, `pmc7615649.xml`, `y5k.html`).

## Sources consulted

- `/home/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/messner2023.py` : the loader (docstring lines 5-54, `process()` 214-315, `create_experiment()` 323-383, `_gene_from_filename()` 386-400) : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/notes/torchcell.datasets.scerevisiae.messner2023.md` : paired note, sections 2026.07.12 and 2026.09.23 : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/notes/tests.torchcell.datasets.scerevisiae.test_messner2023.md` and `tests/torchcell/datasets/scerevisiae/test_messner2023.py` : hermetic tests + their "Findings" section : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/notes/torchcell.datamodels.media-components.md` (Messner rows, lines 128-179) and `torchcell/datamodels/media.py` (`SM`, `_MESSNER_*_QUOTE`, lines 1493-1607) : the media object the loader emits : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/torchcell/verification/runners.py` lines 653-667 : verifier spec (`expected_count 4699`, `allow_duplicate_orfs`) : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/notes/paper.supported-datasets-and-databases.md` line 121 : the paper table row "Messner 2023 (proteome) | 4,699 | 1 | 4,699 | protein abundance | vector (1830)" : read 2026-09-29
- `/home/michaelvolk/Documents/projects/torchcell/notes/experiments.019-simb-multimodal.proteome-expression-eda.md` lines 25-58 : downstream description of `protein_abundance` : read 2026-09-29
- Memories: `no-mendeley-data-source.md` (the Mendeley exception), `metabolic-papers-ocr-capture-status.md` lines 73-93 ("MESSNER LANDED ... 4,699 KO x 1,850 proteins"; historical "4,699 KO x ~2,520 proteins"), `simb2026-multimodal-cgt-plan.md` line 29 ("4699 KO x 1667 proteins" served), `019-v14-proteome-round-cabbi.md` ("1,850-key union", "log2 ratio to HIS3 at build time") : read 2026-09-29
- `/scratch/projects/torchcell-scratch/torchcell-library/messnerProteomicLandscapeGenomewide2023/` : `paper.md` (MinerU OCR, sha256 `edd0fe28...`), `paper.pdf`, `manifest.json`, `data/yeast5k_noimpute_wide.csv`, `data/yeast5k_metadata.csv`; there is NO `si/` directory : read 2026-09-29
- `/scratch/projects/torchcell-scratch/data/torchcell/proteome_messner2023/` : `processed/lmdb` (4,699 entries), `preprocess/data.csv`, `preprocess/verification_report.json`, `preprocess/build_manifest.json` (built 2026-09-24T00:17 UTC at commit 875bb433, `torchcell_dirty: true`) : read 2026-09-29
- `/scratch/projects/torchcell-scratch/data/sgd/genome/*/saccharomyces_cerevisiae_*.gff` : the UniProt cross-reference source : grepped 2026-09-29
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7615649/fullTextXML : Europe PMC author manuscript of the paper (EMS193941); used to cross-check the OCR : fetched 2026-09-29
- https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=37080200 : PubMed record : fetched 2026-09-29
- https://data.mendeley.com/public-api/datasets/w8jtmnszd9?version=1 and `.../w8jtmnszd9/files?folder_id=root&version=1` : Mendeley record + 15-file listing with sha256 : fetched 2026-09-29
- https://data.mendeley.com/public-files/datasets/w8jtmnszd9/files/<id>/file_downloaded for `summary_fileupload.pdf` (63,818 B, sha256 `850c339f...`) : the depositors' file descriptions : fetched 2026-09-29 to the scratchpad only
- https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD036062&outputMode=XML : ProteomeXchange record : fetched 2026-09-29
- https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD036062 : PRIDE API : fetched 2026-09-29 (response: "The project accession is not in the database: PXD036062")
- https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1/datasets/MSV000090136 and https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=8e53ff207e73415a86f4c48a8883ac14 : MassIVE dataset summary + full file browser (12,542 file entries) : fetched 2026-09-29
- https://y5k.bio.ed.ac.uk/ : the Shiny landing page (static HTML only; tab contents render server-side) : fetched 2026-09-29

## Claims

### C1. The curated matrix and metadata come from Mendeley 10.17632/w8jtmnszd9.1 as `yeast5k_noimpute_wide.csv` (sha256 `69a9df05...aa1df9`, 167,754,298 B) and `yeast5k_metadata.csv` (sha256 `48864282...878b`, 377,047 B), mirrored once and hash-pinned
- Previously stated by: loader lines 13-27 and 93-99; note "Data source" section; mirror `manifest.json` `data/` entries
- Verdict: CONFIRMED
- Evidence: Mendeley files API (fetched 2026-09-29) lists `yeast5k_noimpute_wide.csv` size 167754298 sha256_hash `69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9` and `yeast5k_metadata.csv` size 377047 sha256_hash `48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b`. Recomputed on the mirror files (`messner_rawfiles.py`): identical digests. `manifest.json` records `retrieval.method: direct_url`, `retrieved_at 2026-07-12T05:01:38Z`, the exact `public-files` URLs, and `si_data_sources: ["https://data.mendeley.com/datasets/w8jtmnszd9/1"]`. Mendeley record: version 1, `publish_date 2023-04-17`, license CC BY 4.0. `raw/` under the dev root holds hard links (link count 2) of the same two files.
- Consequence: the mirror is the canonical artifact and the upstream bytes have not drifted as of today; the Mendeley URL remains retrieval metadata only.

### C2. The processed protein matrix is Mendeley-only: PRIDE/MassIVE (PXD036062 = MSV000090136) holds only a 74.8 GB raw DIA-NN report; the Cell SI defers to Mendeley; the y5k app has no scriptable download
- Previously stated by: loader lines 13-16; note "Data source" bullets; memory `no-mendeley-data-source.md` EXCEPTION paragraph
- Verdict: PARTLY CONFIRMED
- Evidence:
  - Paper, Data and code availability (OCR line 495-497, identical in Europe PMC XML): "Raw mass spectrometry data have been deposited to the ProteomeXchange Consortium (http://proteomecentral. proteomexchange.org) via the massIVE repository with the dataset identifier ProteomeXchange: PXD036062. ... The measured growth rates and the processed datasets derived from the raw data have been deposited at Mendeley Data and the link is listed in the key resources table. The data are additionally available through an interactive web application: https://y5k.bio.ed.ac.uk/." Key Resources Table (OCR line 481): "Processed proteome data | This study | Mendeley Data: http://doi.org/10.17632/w8jtmnszd9.1".
  - ProteomeXchange XML: `hostingRepository="MassIVE"`, MassIVE id `MSV000090136`, "Unsupported dataset by repository", "Non peer-reviewed dataset", "Dataset with no associated published manuscript" (the PX record was never updated with the Cell citation). PRIDE API: "The project accession is not in the database: PXD036062", so there is no PRIDE-hosted copy at all.
  - MassIVE file browser (task `8e53ff20...`, fetched 2026-09-29): 6,263 `.wiff` + 6,262 `.wiff.scan` raw files, and the non-raw files `search/20210208_diannoutput.tsv` size 74,825,469,404 B (74.8 GB decimal, the DIA-NN precursor report), `library/20210225_specspeclib.tsv` 1,015,662,370 B, `sequence/yeast-uniprot-proteome-3AUP000002311-canonical.fasta` 3,849,759 B, `Description_Methods_Filenaming.pdf`, `Layout_KO_library.ods`, `params.xml`, 9 `.mzML`. No protein matrix (no csv/parquet) is listed. The "74.8 GB" figure is exact.
  - Cell SI: the mirror has no `si/` directory, so Tables S1-S7 were not inspected here. The paper cites Table S1 and S2 as the LC gradient and DIA window tables (OCR line 537), and the KRT routes "Processed proteome data" to Mendeley; the claim "SI defers to Mendeley" rests on the KRT, not on a read of the SI files. Hypothesis (unchecked): no SI table carries the matrix.
  - y5k: the static landing HTML has a nav entry `href="#shiny-tab-download"` labeled "Download" beside "Search Protein", "Search Knock-out", "Methods". Its content is rendered over the Shiny websocket and was not retrievable by a plain fetch, so what the tab serves (per-protein plots, per-KO tables, or the whole matrix) is UNVERIFIABLE here. The claim "no scriptable download" is consistent with what a plain HTTP client sees, but the note's stronger reading ("no download") is not established: the app advertises a Download tab.
- Consequence: the mirror-once + hash-pin exception stands on the PRIDE/MassIVE and KRT evidence; the SI and y5k legs of the argument are weaker than the note implies and should be worded as "the KRT points the processed data to Mendeley; the y5k app exposes a Download tab whose contents were not scriptable".

### C3. The matrix is 1,850 proteins (rows, UniProt `Protein.Group`) x 5,476 samples (columns)
- Previously stated by: loader line 25; note bullet
- Verdict: CONFIRMED
- Evidence: `messner_rawfiles.py`: `matrix shape (1850, 5477) first col Protein.Group`; first ids `A5Z2X5, D6VTK4, O13297, ...`; all 5,476 metadata `Filename`s are matrix columns (389 qc + 4,699 ko + 388 HIS3).

### C4. `yeast5k_metadata.csv` carries `Filename`, `sampletype` (`ko|HIS3|qc`), `ORF`, plate nr
- Previously stated by: loader lines 26-27; note bullet
- Verdict: CONFIRMED (incomplete listing)
- Evidence: columns are `['Filename', 'Injection nr', 'Well nr (counted row-wise)', 'Plate (batch) nr', 'sampletype', 'ORF']`; `sampletype` counts ko 4699, qc 389, HIS3 388; Mendeley `summary_fileupload.pdf`: "Yeast5k_metadata.csv Additional information for each Filename. Column names: Filenames (as in raw data and quant files); Injection nr: Injection number within a batch; Well nr (counted row-wise); Plate (batch) nr; Sampletype; ORF". The note omits `Injection nr` and `Well nr`.

### C5. KO strains are single-replicate: "Strains were not measured in replicates", so `n_replicates = 1` per protein and `protein_abundance_se = None` (P1, P2)
- Previously stated by: loader lines 30-33 and 346-353; note "Single-replicate KOs"
- Verdict: CONFIRMED
- Evidence: paper OCR line 579 (Europe PMC XML identical): "Strains were not measured in replicates. However, for 145 ORFs, more than one strain exists in the library (these strains have different origins)." Limitations (line 240): "our study reports a single proteome per KO strain, and the reported fold-changes are based on relative quantification." Dev LMDB (`messner_lmdb.py`): `se None count 4699`, `n_replicates values (experiment) Counter({1: 4699})`. `protein_abundance_se` is therefore ABSENT for every KO record, neither an SD nor an SE.

### C6. The reference is the 388-replicate WT control (his3D::kanMX complemented by HIS3; `sampletype == HIS3`, `ORF == YOR202W`), measured across 57 batches, aggregated to per-protein mean + SE + n and restricted per record to the strain's measured proteins (P1, P2)
- Previously stated by: loader lines 34-38, 239-257, 354-364; note "388-replicate WT reference"
- Verdict: CONFIRMED
- Evidence: paper line 507: "The control strain (388 replicates) is the complemented his3D deletion strain, haploid from a BY4741 prototrophic deletion collection. This control strain was introduced in 7 positions on each plate ... Plates 56 and 57 contain additional controls." Line 543: "we included 388 WT controls, a strain in which a his3D::kanMX deletion is complemented by heterologous expression of the HIS3 enzyme." Line 575: "taking into account the variation of each protein in the 388 wild-type replicate measurements across the 57 batches." Metadata: 388 HIS3 rows, all `ORF == YOR202W`, spread over all 57 `Plate (batch) nr` values. Every one of the 1,850 proteins has WT coverage (min n = 304). Loader computes `wt.std(ddof=1) / sqrt(n)`, a standard error of the WT mean; dev LMDB reference `n_replicates` range 304-388, no NaN/None SE, and `reference keys == experiment keys` for all 4,699 records. Example (record 0): `YPR010C-A` KO 502.81 vs WT mean 383.22, SE 1.88, n 388.
- Consequence: the served `protein_abundance_se` on the REFERENCE is an SE of the mean over 304-388 WT samples (sample SD, ddof 1); the experiment side carries none.

### C7. 145 ORFs have more than one strain (141 duplicated, 4 triplicated) per the paper; the loader keeps one experiment per KO sample (4,699); case-normalizing `YML009c` makes it 146 (142 + 4)
- Previously stated by: loader lines 39-41; note "Duplicate strains" and "Two data quirks" item 2; verifier `allow_duplicate_orfs`
- Verdict: CONFIRMED
- Evidence: paper line 579: "141 gene deletions are duplicated and 4 triplicated." Metadata, case-sensitive: 145 ORFs with >1 KO sample `{2: 141, 3: 4}`; after uppercasing: 146 `{2: 142, 3: 4}` (raw-case non-matching ORFs are exactly `['YAL043C-a', 'YML009c']`). Dev LMDB: 4,699 records, 4,549 unique KO ORFs, 146 with >1 record `{2: 142, 3: 4}`.

### C8. UniProt accessions map to systematic ORFs through the SGD GFF `protein_id=UniProtKB:` cross-refs, 1,850/1,850, including `P00410 -> Q0250` (COX2); an unmapped id raises
- Previously stated by: loader lines 42-46, 114-142, 221-229; note bullet
- Verdict: CONFIRMED
- Evidence: GFF line `chrmt SGD CDS 73758 74513 ... Parent=Q0250_mRNA;Name=Q0250_CDS;orf_classification=Verified;protein_id=UniProtKB:P00410`. Dev LMDB union of `protein_abundance` keys = 1,850 ORF-form ids; the build would have raised on any miss (loader 223-228). Paper line 563 says the authors used clusterProfiler `bitr` or UniProt for the same conversion; our mapping is independent.

### C9. The stored `protein_abundance` is a LINEAR (not log2) batch-corrected MaxLFQ quantity; "Values ~340-430 confirm linear"; log2 is applied only downstream (P1)
- Previously stated by: loader lines 48-50 and `MEASUREMENT_TYPE = "swath_ms_maxlfq_batch_corrected_quantity"`; note "Value = linear MaxLFQ"
- Verdict: PARTLY CONFIRMED
- Evidence:
  - Mendeley `summary_fileupload.pdf`: "yeast5k_impute_wide.csv: Table contains relative intensity values for each protein in each sample. For details on data processing see 'Normalisation, batch correction, filtering, and protein quantification' in the Methods Section." and "yeast5k_noimpute_wide.csv Same as yeast5k_impute_wide.csv but without imputation (NA values)".
  - Paper line 569-571: DIA-NN 1.7.12; "Precursors were filtered for q-values < 0.01 (precursor and protein level) and only proteotypic peptides were considered. Batches (plates) were corrected by bringing median precursor quantities of each batch to the same value (dividing the quantities by the plate median and multiplying them with the median of all plate medians). Precursors were only considered if identified in > 80% of WT samples and if quantified with CV < 50%. Samples were removed if the number of identified precursors was less than 80% of the maximum number of precursors. Protein quantities were obtained using the MaxLFQ algorithm as implemented in the DIA-NN R package". Line 575: "Differential abundance analysis was conducted on the processed data (see above) after log2 transformation."
  - File measurements (`messner_rawfiles.py`, `messner_batch.py`): all values positive, min 0.0313, median 235.1, max 393,221; quantiles 1%/99% = 27.0 / 13,975. Per-plate medians of sample medians range 228.8-261.5 (CV 3.0% across 57 plates), consistent with plate-median correction. Recomputed median protein CV on the linear values (SD/mean, as the paper defines CV at line 559): WT 11.0% (paper: 11.3%), QC 7.9% (paper: 8.1%), KO 15.5% (paper: 16.2%). These match the paper's Figure 1D numbers closely enough to identify the file as the paper's filtered, linear-scale dataset.
  - What is REFUTED inside the claim: the note's "Values ~340-430 confirm linear" misdescribes the data; the values span four orders of magnitude (0.03 to 3.9e5) and the median is 235. Also "batch-corrected MaxLFQ" compresses the order of operations: batch correction is applied to PRECURSOR quantities before MaxLFQ; the protein values are MaxLFQ outputs of batch-corrected precursors. The number is a relative label-free intensity (the depositors' own word is "relative"), not a ratio to WT and not log2.
- Consequence: the served scalar is safe to describe as "linear-scale MaxLFQ relative protein intensity from plate-median-corrected precursors, no imputation"; the measurement_type string is defensible but the docstring and note sentence should be corrected.

### C10. The no-imputation matrix is used so every stored value is measured; NaN dropped per strain; first strain stored 1,830/1,850
- Previously stated by: loader lines 22-24, 269; note "noimpute over impute"
- Verdict: CONFIRMED
- Evidence: Mendeley readme (C9) defines noimpute as the un-imputed table. KO cells NaN fraction 2.61% (HIS3 2.25%, qc 1.75%); per-record measured proteins 1,441-1,849 (median 1,819), sum 8,466,210 = `preprocess/data.csv` first row `n_proteins 1830` and LMDB `proteins per record min/median/max/sum 1441 1819 1849 8466210`. The paper's imputation rule (line 571, mixed random-min / KNN) is exactly what is avoided.

### C11. Background = S288C BY4741 MATa deletion collection with restored prototrophy (Mulleder); KO modeled as `KanMxDeletionPerturbation`; the prototrophy marker is not modeled
- Previously stated by: loader lines 7-9, 50-52, 329-341; note "Background"
- Verdict: PARTLY CONFIRMED
- Evidence: paper line 507: "We measured proteomes for all strains of Saccharomyces cerevisiae (S288c) haploid (MATa) deletion collection5 with restored prototrophy94 that could be cultivated without major growth defect in minimal dextrose medium." Ref 5 = Winzeler 1999 (kanMX deletion collection); ref 94 = Mulleder et al. 2012 Nat Biotechnol 30:1176 (the prototrophic collection). KRT: "Prototrophic Saccharomyces cerevisiae deletion collection (MATa, restored prototrophy) | Winzler et al.5; Mülleder et al.94 | http://www.euroscarf.de/". The word "BY4741" appears for the control ("haploid from a BY4741 prototrophic deletion collection") and for the SGA-background construction (line 533), not in the KO-collection sentence itself; the BY4741 identity of the KO strains is inherited from the Winzeler/Mulleder collections. The kanMX cassette for the KO strains is likewise sourced from the collection, this paper naming kanMX only for the control ("his3D::kanMX"). Dev LMDB: `strain BY4741`, `ploidy haploid`, `perturbation_type kanmx_deletion` for all 4,699. The restored-prototrophy element (Mulleder 2012's marker complementation) and the control's HIS3 complementation are not in the genotype, as the note already records.
- Consequence: correct as far as it goes; the showcase should attribute BY4741 and kanMX to the collection (refs 5, 94) rather than to a Messner sentence, and keep the prototrophy gap flagged.

### C12. Medium = synthetic minimal (SM) liquid, 6.7 g/L YNB without amino acids + 2% glucose, 30 C; the protocol defers to Mulleder 2016 (ref 15) and restates the recipe (P3)
- Previously stated by: loader lines 52-53, 342-345; note 2026.09.23 section; `media.py` `_MESSNER_*_QUOTE`
- Verdict: CONFIRMED
- Evidence: paper line 77: "We grew a prototrophic derivative of the yeast gene deletion collection in a synthetic minimal (SM) medium without amino acid and nucleobase supplementation"; line 513: "The yeast strains were grown as previously published15 with slight modifications. The thawed stock cultures were spotted with the pinning robot onto SM agar medium (6.7 g/l yeast nitrogen base without amino acids, 2% glucose, 2% agar) and incubated at 30°C for 47–49 hours. Subsequently, these cells were used for inoculation in 200 μl SM liquid medium in 96-well plates and incubated at 30°C. After 19.75 hours, 160 μl culture was transferred to a deep-well plate ... pre-filled with 1,440 μl SM liquid medium (1/10 dilution) ... incubated for 8 hours at 30°C with 1,000 rpm mixing" (Europe PMC rendering; the OCR line 513 carries the same text with LaTeX artifacts). Ref 15 (OCR line 328) = Mulleder 2016 Cell 167:553. Limitations (line 238): "we chose a minimal medium and a prototrophic background". Dev LMDB: media name "SM (synthetic minimal: 6.7 g/L YNB without amino acids + 2% glucose), liquid", temperature 30.0 C, aerobic, for all 4,699. KRT prints the YNB catalog number as `Cat#Y0262` (media note already records the Y0626 transposition).
- Growth phase: the paper gives no OD or phase word for the harvest; it is 8 h after a 1/10 dilution of a 19.75 h preculture. The loader stores `duration_hours = None` and no phase (see N4).

### C13. L0-L4 verification passes: L0 4,699; L1 count 4,699; L1 orf_uniqueness 4,549 ORFs / 146 multi-strain; L2 value_fidelity 8,466,210; L2 se_nonnegative 0 values; L3 reference_finite all 8,466,210; L3 measurement_type single
- Previously stated by: note "L0-L4 verification"
- Verdict: CONFIRMED
- Evidence: `preprocess/verification_report.json` (written 2026-09-23 19:21) reports exactly those numbers, plus an L4 `gene_containment_scmd_ohya2005` pass (4,300 / 4,549 = 0.945). `build_manifest.json`: built 2026-09-24T00:17 UTC on gilahyper at commit 875bb433, `torchcell_dirty: true`.

### C14. Non-systematic KO ORFs are skipped; inconsistent casing (`YML009c`, `YAL043C-a`) is normalized rather than dropped, giving the full 4,699
- Previously stated by: loader lines 263-268; note "Two data quirks" item 1
- Verdict: CONFIRMED
- Evidence: after uppercasing, zero KO ORFs fail `^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$`; case-sensitively exactly `YAL043C-a` and `YML009c` would fail. LMDB holds 4,699 records, one `kanmx_deletion` each.

### C15. Paper totals: 4,699 KOs, 1,850 proteins, 8,693,150 protein quantities; dev LMDB 4,699 records x 1,850-protein union (P5)
- Previously stated by: loader line 147 and docstring; note; memory `metabolic-papers-ocr-capture-status.md` ("4,699 KO x 1,850 proteins")
- Verdict: PARTLY CONFIRMED
- Evidence: paper line 79: "This map contains more than 100 million peptide quantities mapped to 8,693,150 protein quantities, providing information on 1,850 unique proteins across the 4,699 measured KOs"; line 79 also: "the average quantification of 2,520 proteins per sample. In total, 3,205 proteins were measured in at least 10% of the samples" (pre-filter). Dev LMDB: 4,699 records; union 1,850 proteins; 8,466,210 stored values. 4,699 x 1,850 = 8,693,150 exactly, so the paper's "8,693,150 protein quantities" is the full KO x protein grid INCLUDING imputed cells; the noimpute store carries 8,466,210 measured values (97.4%). The older memory line "4,699 KO x ~2,520 proteins" describes the unfiltered per-sample average, not the released matrix.

### C16. Publication: PMID 37080200, doi 10.1016/j.cell.2023.03.026, Cell 186:2018
- Previously stated by: loader lines 7 and 377-382; note header
- Verdict: CONFIRMED
- Evidence: PubMed esummary: "The proteomic landscape of genome-wide genetic perturbations. | Cell 186 2018-2034.e21 2023 Apr 27", doi `10.1016/j.cell.2023.03.026`, PMCID `PMC7615649` (author manuscript EMS193941). The task text's PMC guess (PMC10112277) is a different article (a NeuroImage paper); the correct open-access copy is PMC7615649.

### C17. The paper table lists the Messner label as "vector (1830)"
- Previously stated by: `notes/paper.supported-datasets-and-databases.md` line 121
- Verdict: PARTLY CONFIRMED
- Evidence: 1,830 is the FIRST record's measured-protein count (`data.csv` row 1). Per-record counts range 1,441-1,849 and the union is 1,850. A fixed "1830" understates the dimensionality and overstates uniformity.

### C18. The served graph holds the Messner proteome as "4699 KO x 1667 proteins" (memory `simb2026-multimodal-cgt-plan.md`) versus "1,850-key union" (memory `019-v14-proteome-round-cabbi.md`)
- Previously stated by: the two memories
- Verdict: UNVERIFIABLE (no Neo4j access in this task); internally inconsistent
- Evidence: dev LMDB union is 1,850. The 1,667 figure has no recorded derivation; hypothesis (untested): it was a count over a subset (proteins present in every record, or a Kemmeren-joined set) rather than the union.

### C19. Test note: `_gene_from_filename` returns the token after the ORF unvalidated, "so this only bites on a malformed filename"
- Previously stated by: `notes/tests.torchcell.datasets.scerevisiae.test_messner2023.md` "Findings"
- Verdict: REFUTED
- Evidence: `preprocess/data.csv` (the loader's own `gene` column, which becomes `perturbed_gene_name`): 156 records carry a purely numeric gene token, from real filenames such as `10_9_hpr57_ko_YBL007C_2824_0.49`, `100_93_hpr56_ko_YDR230W_3705_0.31`, `13_12_hpr56_ko_YLR069C_1589_NA`, `14_13_hpr56_ko_YBR196C-A_3533_0.19` (154 distinct ORFs, concentrated at wells hpr56/hpr57). 133 of those 154 ORFs have an SGD standard name (e.g. YBL007C = SLA1, YDL149W = ATG9, YFL036W = RPO41, YDL104C = QRI7), so the served `perturbed_gene_name` for SLA1 is the string "2824". Separately, 1,182 records store a lowercase ORF (e.g. `YBR174c`, `YLR169w`) as the gene name and 30 fall back to the ORF exactly; 4 carry punctuation (`MF(ALPHA)1`, `DUR1,2`, `IMP2'`, `MF(ALPHA)2`).
- Consequence: `perturbed_gene_name` is unreliable for 156 + 1,182 records; the systematic name (`systematic_gene_name`) is correct throughout, so joins by ORF are unaffected, but any display or gene-name join using `perturbed_gene_name` is wrong for these records.

### C20. The verifier's provenance method string ("plate-median batch-corrected, no imputation; single-replicate KOs vs a 388-replicate HIS3 WT reference; UniProt->ORF via SGD GFF")
- Previously stated by: `runners.py` lines 660-664
- Verdict: CONFIRMED
- Evidence: each clause verified under C5, C6, C8, C9, C10.

## New findings not previously recorded

- N1 (data quality, served). 156 records' `perturbed_gene_name` is an integer string from the filename (see C19); 133 of the 154 affected ORFs have SGD standard names. Nothing in the note, loader, or verifier flags it. The fix is to derive `perturbed_gene_name` from SGD (the GFF already loaded for UniProt mapping carries `gene=`), never from the filename.
- N2 (data quality, served). 1,182 records store a lowercase ORF as `perturbed_gene_name` (the filename convention for genes with no standard name), and 30 the exact ORF; three spellings for "no standard name". Same fix as N1.
- N3 (paper number semantics). The paper's "8,693,150 protein quantities" equals 4,699 x 1,850 and therefore counts imputed cells; the un-imputed store has 8,466,210 (97.4%). Any showcase text quoting 8.69 M must say it is the imputed grid.
- N4 (environment completeness). The harvest timeline is fully stated (47-49 h on SM agar, 19.75 h in 200 µl SM liquid, 1/10 dilution, 8 h at 30 C, 1,000 rpm) but no OD or growth-phase word is given; the loader stores `duration_hours = None`. The 8 h post-dilution duration is a sourceable value that is currently not recorded.
- N5 (repository facts). PXD036062 is not in PRIDE's database (MassIVE-hosted, "Unsupported dataset by repository", never linked to the Cell citation in ProteomeXchange). MassIVE MSV000090136 holds 6,263 `.wiff` raw files, `search/20210208_diannoutput.tsv` (74,825,469,404 B), `library/20210225_specspeclib.tsv` (1,015,662,370 B), the UniProt canonical FASTA (UP000002311), `Description_Methods_Filenaming.pdf`, `Layout_KO_library.ods`, `params.xml`. The note's "74.8 GB" is exact; the additional files (spectral library, plate layout, file-naming description) were not recorded and the layout file is a second, independent source for the KO-to-well assignment.
- N6 (y5k). The app's nav has a "Download" tab (`#shiny-tab-download`); its contents are server-rendered and were not inspected. The note's "no scriptable download" should not be read as "no download".
- N7 (Mendeley readme). The depositors describe the matrix as "relative intensity values" and defer all processing detail to the Methods; the record lists 15 files including `yeast5k_impute_wide.csv` (171,281,957 B), `yeast5k_stat_DE.csv`, `yeast5k_stat_DE_bySTRAIN.csv` (the readme calls it `_byORF`), `yeast5k_growthrates_byORF.csv`, and the PPS/CRG score tables; only the two used files are mirrored.
- N8 (note accuracy). "Values ~340-430 confirm linear" is wrong as a description of the matrix (range 0.03-393,221, median 235); linear scale is instead confirmed by the CV reproduction (WT 11.0% vs 11.3%, QC 7.9% vs 8.1%, KO 15.5% vs 16.2%).
- N9 (verifier message). The L4 message reads "0.945 of scmd_ohya2005's 4549 deletion genes are in Ohya"; 4,549 is Messner's ORF count, so the sentence has its subjects swapped.
- N10 (build hygiene). Beside the live build sit `processed.stale-messner/`, `preprocess.stale-messner/` (uid 7474 files) and `deprecated-stale-2026-09-14/` (1.1 GB `experiment_reference_index.json`); the live `build_manifest.json` records `torchcell_dirty: true`, so the dev LMDB was built from an uncommitted tree at 875bb433.
- N11 (PMCID). The open-access full text is PMC7615649 (Europe PMC author manuscript), not PMC10112277.
- N12 (memory inconsistency). Served protein count is recorded as 1,667 in one memory and 1,850 in another (C18); neither cites a query.

## What the public showcase page can state safely

- Source paper: Messner et al. 2023, Cell 186:2018-2034.e21, PMID 37080200, doi 10.1016/j.cell.2023.03.026 (PubMed esummary); open-access author manuscript PMC7615649.
- Design: proteomes of 4,699 single-gene deletion strains of the S288c haploid MATa deletion collection with restored prototrophy (paper line 507, refs 5 and 94), grown in synthetic minimal medium (6.7 g/L YNB without amino acids, 2% glucose, no amino acid or nucleobase supplement) at 30 C (paper lines 77, 513), measured by microflow LC on a TripleTOF 6600 with DIA/SWATH and processed with DIA-NN 1.7.12 (paper lines 537, 569).
- Replication: "Strains were not measured in replicates" (paper line 579); the wild-type control (his3D::kanMX complemented by HIS3) was measured 388 times across 57 plates (paper lines 507, 543, 575), and 389 pooled-digest QC injections were run.
- Stored quantity: linear-scale MaxLFQ relative protein intensity from plate-median-corrected precursors, without imputation (Mendeley readme; paper line 571); per-KO `n_replicates = 1`, no per-KO SE; the reference carries the WT mean, SE of the mean (sample SD / sqrt n) and n (304-388) per protein, restricted to the proteins that strain measured.
- Size: 4,699 records over 4,549 deletion ORFs (146 ORFs with 2-3 independent strains, kept separately as the paper does for its descriptive analysis), 1,850 proteins in the union, 8,466,210 measured values (the paper's 8,693,150 is the imputed grid).
- Provenance: `yeast5k_noimpute_wide.csv` sha256 `69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9` and `yeast5k_metadata.csv` sha256 `48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b`, fetched once from Mendeley Data 10.17632/w8jtmnszd9.1 (CC BY 4.0, published 2023-04-17) on 2026-07-12 and pinned in the mirror; raw data at MassIVE MSV000090136 / ProteomeXchange PXD036062 (raw `.wiff` plus a 74.8 GB DIA-NN report, no protein matrix).
- Do NOT state on the showcase until fixed: gene names from `perturbed_gene_name` (N1, N2); "1830 proteins per strain" (C17); any served protein count (C18).

## Totals
CONFIRMED 13, REFUTED 1, PARTLY 5, UNVERIFIABLE 1
