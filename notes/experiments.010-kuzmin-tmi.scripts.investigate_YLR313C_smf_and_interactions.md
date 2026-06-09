---
id: j8g7eagfq414svyhp44qvsd
title: investigate_YLR313C_smf_and_interactions
desc: ''
updated: 1781029041289
created: 1781029041289
---

## 2026.06.09 - Deciding Whether to Swap the Merged ORF YLR312C-B for the Real Gene SPH1

This investigation exists to inform a node-swap decision in the inference-3 panel: YLR312C-B is an SGD "Merged" small ORF that does not encode a discrete protein and is now an alias of SPH1 / YLR313C, whose CDS contains the old ORF span -- so a full-ORF KanMX replacement of YLR312C-B actually deletes SPH1. The script gathers single- and double-mutant phenotypes for both nodes, plus the model's triple-interaction predictions, so the team can judge whether the model's feature node should be replaced by the authentic gene and whether the predicted interactions are biologically real.

### Specifics worth keeping

- Singles come from the small Smf LMDB datasets; doubles (epsilon, P-value, DMF) are streamed from each study's flat `preprocess/data.csv` with a substring pre-filter, deliberately avoiding deserialization of the 20.7M-record LMDB.
- Significance is judged with P < 0.05 and |epsilon| > 0.08; only Costanzo2016 carries epsilon and P-values.
- Outputs: `YLR313C_investigation_singles_queried.csv`, `YLR313C_investigation_doubles_queried.csv`, `YLR313C_investigation_costanzo_genomewide.csv`, a triple-prediction scatter/histogram plot and a genome-wide Costanzo epsilon histogram (both in `assets/images/010-kuzmin-tmi/`), and a markdown table snippet printed to stdout for this note.
