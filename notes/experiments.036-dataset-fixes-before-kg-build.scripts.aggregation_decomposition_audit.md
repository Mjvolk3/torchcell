---
id: 48jvitgxk52hr4ylfwgk8gc
title: Aggregation_decomposition_audit
desc: ''
updated: 1791694258092
created: 1791694258092
---

## 2026.10.11 - What the audit measures and how to re-run it

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py`. Policy and findings: [[torchcell.candidates.aggregation-decomposition]] (plan decision 16).

For each served bacterial compendium (Price 2018 Keio, Borchert 2024, Lim 2022, Lamoureux 2023 PRECISE-1K and Public K-12) it lists the source studies from the compendium's attribution column (and, for RB-TnSeq, the Fitness Browser's `pubId`), then per source: the own release (kind, URL, scriptable or walled), the mirror key, whether a registered dataset serves it, compendium and served sample counts, and the L4 cross-check where both sides are in hand.

Inputs, read-only: the raw and literature mirrors, the dev stores' `preprocess/` ledgers (never an LMDB), and the Fitness Browser snapshot `feba.db` (sha256 `250d4a03...`, 10,205,732,864 bytes, pinned in the script and re-hashed before it is opened).

```bash
# from the repo root
python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py --feba-db <feba.db> --network   # re-probe every release URL
python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py --feba-db <feba.db> --reprobe-failed  # re-ask 5xx probes only
python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py --tables-only   # re-render the note tables
python experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py --yeast-only    # YEASTRACT+ attribution, SPELL on disk
python -m torchcell.candidates gate --row "Borchert 2024 fModules" --phenotype-class EnvironmentResponsePhenotype \
    --decomposition experiments/036-dataset-fixes-before-kg-build/results/aggregation_decomposition_audit.json --write
```

Outputs in `experiments/036-dataset-fixes-before-kg-build/results/`: `aggregation_decomposition_audit.json` (its `decompositions` key is the CLI contract), `aggregation_decomposition_probes.json` (250 probes, 2026-10-11), `aggregation_decomposition_tables.md`, `aggregation_decomposition_yeast_sources.json`.

Choices made here:

- A transient probe (5xx, or a raised request) stops the audit rather than reading as "no release"; ENA answered 500 to 16 read-run reports on the first pass and 200 on the re-ask.
- A file already in the raw mirror takes its access from the manifest's recorded retrieval (`manual_browser` is walled, anything else scriptable), not from a probe with this script's User-Agent.
- GEO per-record values are judged by supplementary file names (tracks, images and sequences excluded); nothing was opened. A name can read as a contrast rather than per-sample values (Bordbar 2014's `GSE48324_CRA_v_CRA+trp.csv.gz`) and still count as per-record values by this rule; each loader issue's first step is to open the file.
- A browser group the compendium already attributes takes that study's pinned DOI; otherwise the browser publication is matched to a mirror key by title, then by the DOI in its URL (a preprint DOI when that is all the browser links).
- The mirror index reads `citation_key`, `doi` and `title` off every key's manifest and lists the ones the current `Manifest` model refuses (`torchcell-raw/zhangCombiningMechanisticMachine2020`, already #923).
