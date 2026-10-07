---
id: p4fcuxu1kjwgt1u67q4e6na
title: Test_loader_mains
desc: ''
updated: 1791361456292
created: 1791361456292
---

## 2026.10.07 - Origin (Phase 24 of the test campaign)

The loader `main()` functions are build entry points: `experiments/database/datasets.sh` runs them, so their contract is which tree they build into and with what, not what they print for a human. Phase 23 kept them in the library for that reason; this file pins the contract without a build.

Fixture: `DATA_ROOT` is a `tmp_path` directory (four mains read the drop log under the root, so a sentinel string is not enough); `dotenv.load_dotenv` is a recording no-op so the repo `.env` cannot override the root and the test can pin that every main loads it exactly once; `SCerevisiaeGenome` and each dataset class are replaced on the loader module by recording fakes whose two hand-built records give `len = 2` and a fixed `dataset[0]`.

Pinned, per loader (16 modules plus `sgd_gene_graph`): the exact `root=$DATA_ROOT/data/torchcell/<name>` string of every dataset class the main constructs, in order; the genome kwargs `genome_root=$DATA_ROOT/data/sgd/genome`, `go_root=$DATA_ROOT/data/go`, `overwrite=False` where a genome is built, and that the same genome object reaches every dataset; that `vanacloig2022` and `wildenhain2015` never construct a genome; the exact stdout of each main (the `len = 2` line, the first record, Bloom's `cross ... segregant ... blocks 3` and `phenotype 0.25` lines, Mormino's `json.dumps` of the experiment, the `rules` JSON of the four drop-log mains and Costanzo 2021's `n_dropped_records`). For `sgd_gene_graph` the genome is dropped of chrmt then empty GO, the graph receives the sgd, string and tflink roots and the genome, and one dataset is built per `MODEL_TO_WINDOW` key with exact kwargs.

Audit (Opus 5.5, read-only): 21 accept, 3 notes, no rewrite or reject across Lane C; the notes added the `load_dotenv` call pin and the TMM reference control in the Vanacloig test. Record in [[test-campaign.2026.09.25]].
