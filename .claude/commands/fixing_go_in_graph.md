## Project Overview

We're working on improving the Gene Ontology (GO) term handling in TorchCell,
specifically focusing on properly integrating GO annotations from multiple sources
within the graph representation module. The goal is to ensure that the G_go graph in
TorchCell correctly incorporates GO terms from both raw SGD (Saccharomyces Genome
Database) data and GAF (Gene Association File) data.

## Key Files

1. /Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py - Core file
containing the graph representation classes and GO term handling
2. /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/script
s/go_vs_go_raw.py - Original script for comparing GO annotations (has issues)
3. /Users/michaelvolk/Documents/projects/torchcell/torchcell/sequence/genome/scerevisiae/s288c.py

## Main Issues Identified

1. GAF Data Loading Issue: The original implementation wasn't properly using the goatools library to parse GAF files. This resulted in missing GO terms and annotations.
2. Incorrect Data Structure Handling: The GAF data structure was being accessed incorrectly in the analysis scripts, leading to zero GAF annotations being extracted
despite the data being loaded.
3. Data Integration Gap: The extracted GO terms from the GAF file weren't properly being integrated into the G_go graph, resulting in the graph missing important annotations.

## Supporting Information

- We have 37,854 go annotations as previously evaluated whereas there are 120,735 reported by [geneontology.org](https://current.geneontology.org/products/pages/downloads.html) - This is a pretty big discrepancy that probably accounts for the differences we are seeing.

```python
sum([len(v) for k,v in graph.go_to_genes.items()])
67372
```

## Current Issues

Now filtering by evidence code IGI is not working. 


## sgd.gaf

```bash
!gaf-version: 2.2
!
!generated-by: GOC
!
!date-generated: 2025-03-18T20:15
!
!Header from source association file:
!=================================
!
!generated-by: GOC
!
!date-generated: 2025-03-17T16:49
!
!Header from sgd source association file:
!=================================
!date-generated: 20250310
!generated-by: Saccharomyces Genome Database (SGD)
!URL: https://www.yeastgenome.org/
!Contact Email: sgd-helpdesk@lists.stanford.edu
!Funding: NHGRI at US NIH, grant number U41-HG001315
!
!=================================
!
!Header copied from paint_sgd_valid.gaf
!=================================
!Created on Thu Nov 21 14:20:18 2024.
!generated-by: PANTHER
!date-generated: 2024-11-21
!PANTHER version: v.19.0.
!GO version: 2024-11-03.
!
!=================================
!
!Documentation about this header can be found here: https://github.com/geneontology/go-site/blob/master/docs/gaf_validation.md
!
SGD	S000003381	GPC1	acts_upstream_of_or_within	GO:0090640	PMID:30514764	IGI	SGD:S000005701	P	Glycerophosphocholine acyltransferase (GPCAT)	YGR149W|glycerophosphocholine acyltransferase	protein	taxon:559292	20181220	SGD		UniProtKB:P48236
SGD	S000005701	ALE1	acts_upstream_of_or_within	GO:0090640	PMID:30514764	IGI	SGD:S000003381	P	Broad-specificity lysophospholipid acyltransferase	YOR175C|SLC4|LPT1|LCA1|lysophospholipid acyltransferase	protein	taxon:559292	20181220	SGD		UniProtKB:Q08548
SGD	S000003381	GPC1	acts_upstream_of_or_within	GO:0036151	PMID:30514764	IMP		P	Glycerophosphocholine acyltransferase (GPCAT)	YGR149W|glycerophosphocholine acyltransferase	protein	taxon:559292	20181220	SGD		UniProtKB:P48236
SGD	S000004492	RCF1	acts_upstream_of_or_within	GO:0033617	PMID:29746825	IMP		P	Cytochrome c oxidase subunit	YML030W|AIM31|respiratory supercomplex assembly factor RCF1	protein	taxon:559292	20181212	SGD		UniProtKB:Q03713
SGD	S000004977	SIW14	enables	GO:0052845	PMID:26828065	IDA		F	Inositol phosphatase involved in inositol pyrophosphate metabolism	YNL032W|OCA3|putative tyrosine protein phosphatase SIW14	protein	taxon:559292	20190110	SGD	part_of(GO:0071543)	UniProtKB:P53965
SGD	S000004977	SIW14	involved_in	GO:0071543	PMID:26828065	IMP		P	Inositol phosphatase involved in inositol pyrophosphate metabolism	YNL032W|OCA3|putative tyrosine protein phosphatase SIW14	protein	taxon:559292	20190110	SGD		UniProtKB:P53965
SGD	S000000967	PAB1	acts_upstream_of_or_within	GO:0031124	PMID:9223284	IMP		P	Poly(A) binding protein	YER165W|polyadenylate-binding protein	protein	taxon:559292	20190114	SGD		UniProtKB:P04147
SGD	S000001293	ULP2	located_in	GO:0000785	PMID:30575729	IDA		C	Peptidase that deconjugates Smt3/SUMO-1 peptides from proteins	YIL031W|SMT4|SUMO protease ULP2	protein	taxon:559292	20190130	SGD		UniProtKB:P40537
SGD	S000001912	AIP5	acts_upstream_of_or_within	GO:0032233	PMID:31699995	IDA		P	Protein that positively regulates actin assembly	YFR016C	protein	taxon:559292	20191113	SGD		UniProtKB:P43597
SGD	S000001912	AIP5	located_in	GO:0005934	PMID:31699995	IDA		C	Protein that positively regulates actin assembly	YFR016C	protein	taxon:559292	20191113	SGD		UniProtKB:P43597
SGD	S000001912	AIP5	located_in	GO:0005935	PMID:31699995	IDA		C	Protein that positively regulates actin assembly	YFR016C	protein	taxon:559292	20191113	SGD		UniProtKB:P43597
SGD	S000005222	CAF120	located_in	GO:0005935	PMID:25028499	IDA		C	Part of the CCR4-NOT transcriptional regulatory complex	YNL278W	protein	taxon:559292	20191114	SGD		UniProtKB:P53836
SGD	S000003211	ANK1	is_active_in	GO:0005737	PMID:37531259	IDA		C	Cytoplasmic ankyrin repeat-containing protein	YGL242C|ankyrin repeat-containing protein ANK1	protein	taxon:559292	20230809	SGD		UniProtKB:P53066
```
