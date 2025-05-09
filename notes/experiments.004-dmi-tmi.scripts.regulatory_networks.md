---
id: jkyl52kkqr7972fuii0c5z3
title: Regulatory_networks
desc: ''
updated: 1745553891244
created: 1745553887144
---
```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/experiments/004-dmi-t
mi/scripts/regulatory_networks.py
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
Total regulatory edges: 9753

Sources of regulatory interactions:

Annotation types for regulatory interactions:
high-throughput: 9216 interactions (94.5%)
manually curated: 537 interactions (5.5%)

Experiment types for regulatory interactions:
chromatin immunoprecipitation-chip evidence: 5807 interactions (59.5%)
DNA to cDNA expression microarray evidence: 2592 interactions (26.6%)
quantitative mass spectrometry evidence: 299 interactions (3.1%)
RNA-sequencing evidence: 246 interactions (2.5%)
quantitative reverse transcription polymerase chain reaction evidence: 142 interactions (1.5%)
chromatin immunoprecipitation- exonuclease evidence: 119 interactions (1.2%)
chromatin immunoprecipitation-seq evidence: 107 interactions (1.1%)
genomic systematic evolution of ligands by exponential amplification evidence: 76 interactions (0.8%)
protein kinase assay evidence: 69 interactions (0.7%)
qualitative western immunoblotting evidence: 62 interactions (0.6%)

Detailed data for 5 sample regulatory edges:

Edge 1: YIR018W → Q0050
  Reference: Pimentel C, et al. (2012), PubMed ID: 22616008
  Experiment: DNA to cDNA expression microarray evidence
  Regulation type: transcription
  Annotation type: high-throughput

Edge 2: YIR018W → Q0080
  Reference: Pimentel C, et al. (2012), PubMed ID: 22616008
  Experiment: DNA to cDNA expression microarray evidence
  Regulation type: transcription
  Annotation type: high-throughput

Edge 3: YIR018W → Q0120
  Reference: Pimentel C, et al. (2012), PubMed ID: 22616008
  Experiment: DNA to cDNA expression microarray evidence
  Regulation type: transcription
  Annotation type: high-throughput

Edge 4: YIR018W → YAL002W
  Reference: Venters BJ, et al. (2011), PubMed ID: 21329885
  Experiment: chromatin immunoprecipitation-chip evidence
  Regulation type: transcription
  Annotation type: high-throughput

Edge 5: YIR018W → YAL011W
  Reference: Venters BJ, et al. (2011), PubMed ID: 21329885
  Experiment: chromatin immunoprecipitation-chip evidence
  Regulation type: transcription
  Annotation type: high-throughput
```
