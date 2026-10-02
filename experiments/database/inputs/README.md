# `experiments/database/inputs/`

Hand-delivered inputs to the database-expansion curation scripts. Everything here is
an **input**, not an output, which is why this directory is tracked while
`../results/` is not: the files under `results/` are regenerable by rerunning a
script, and these are not regenerable by anything.

Each file is pinned by sha256 in the script that reads it, and the script fails rather
than warns on a mismatch. A file that arrives as a file rather than from a URL has no
retrieval command, so the stored copy is its only canonical form and losing it is
unrecoverable.

| file | read by | pin |
|---|---|---|
| `ecoli-putida-300-discovery-queue.xlsx` | `../scripts/build_bacteria_discovery_queue.py` | `XLSX_SHA256` |

## `ecoli-putida-300-discovery-queue.xlsx`

A 300-publication candidate sweep for *E. coli* and *P. putida*, 200 and 100 rows,
received 2026-10-02. Its own Overview sheet records how it was made, "Candidate
discovery used Europe PMC publication metadata/API, followed by rule-based scope
filtering and prioritization", and states its own limit, "this is a 300-candidate
discovery queue, not 300 download-verified datasets."

No row carries a verified record count: every cell of its `Instances` column reads
"Verify from supplement". Promoting a row into the curated table in
`../scripts/build_bacteria_candidate_datasets_table.py` means fetching the paper and
counting its axes by hand. A queue row must never be cited as though its scale were
known.
