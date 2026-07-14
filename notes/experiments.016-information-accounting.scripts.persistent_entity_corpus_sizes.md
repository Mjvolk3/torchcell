---
id: i1o3pf11hx5pqfkcgxb6d8s
title: Persistent_entity_corpus_sizes
desc: ''
updated: 1783913921382
created: 1783913921382
---

Measures the **persistent-entity** side of Fig. 1c and Supplementary Note 5 by asking each public
archive how large its distributed, compressed artifact is. Backs
[[paper.information-accounting]]; the Note itself lives in
`paper/nature-biotech/sections/backmatter.tex`.

- script: `experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py`
- results: `experiments/016-information-accounting/results/persistent_entity_corpus_sizes.{csv,json}` (latest)
- **snapshots (immutable — cite these):** `results/snapshots/persistent_entity_corpus_sizes_<ts>.{csv,json,audit.csv.gz}`
- paper table (AUTO-GENERATED, do not hand-edit):
  `paper/nature-biotech/sections/tab-entity-corpora.tex` → renders as **Table S2**

```bash
# from repo root; ~5 min (PubChem HEADs are rate-limited, wwPDB is 1,243 dir listings)
python experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py --write-table
# reformat the table without re-querying the archives (keeps the original snapshot_id):
python experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py --from-csv --write-table
```

## 2026.07.12 - Citability: what to point at when the numbers are questioned

**The paper's numbers are snapshot `2026-07-12-23-02-43`** (`fetched_utc 2026-07-13T04:02:43Z`).
Table S2's LaTeX header names this snapshot, so the table is self-dating.

The archives are **live and grow monotonically**, so the script is reproducible in *method* but not
in *value*: re-run it next month and the entity total is larger. That is not a weakness — a bigger
entity corpus only widens the Fig-1c gap, so **the published separation is conservative**, and the
table caption says exactly that. What must not happen is losing the measured snapshot, hence:

| Artifact | Role |
| --- | --- |
| `..._<ts>.csv` | the six rows exactly as published |
| `..._<ts>.json` | + every URL queried, each archive's own release id, the method per row, and the caveat text |
| `..._<ts>.audit.csv.gz` | **every per-file size that was summed** — 256,067 rows (357 PubChem blocks + 255,710 wwPDB `.cif.gz`), 1.5 MB |

Four of the six rows are a *single authoritative field* straight from the archive (UniProt
`Content-Length`; BLAST `bytes-total-compressed`). The other two — PubChem and wwPDB — are **sums**,
so the audit file exists to let anyone re-check the arithmetic **offline, with no network**:

```python
audit = pd.read_csv(f"{base}.audit.csv.gz")
audit.groupby("corpus").compressed_bytes.sum()
#   PubChem Compound    115,755,280,231   == table
#   wwPDB mmCIF          89,192,592,160   == table
```

Verified: both re-sum byte-for-byte.

**Archive release ids pinned in the snapshot** (this is what a referee reconciles against, not the
fetch date): UniProt `2026_02`; NCBI `nt` `last-updated 2026-07-08`; RefSeq RNA `2026-06-26`;
PubChem and wwPDB are rolling, so their release *is* the fetch date.

**Retry, not fallback.** `_open()` retries 4× with backoff on DNS faults and 5xx, and re-raises 4xx
immediately and everything after exhaustion. Two runs died mid-way to transient DNS before this was
added; a ~1,600-request run that produces a *citable* number must not be lost to one dropped packet.
This masks nothing — every attempt hits the same authoritative endpoint.

## 2026.07.12 - Why compressed size, and how each archive is read

### The measure

For a corpus $D$, a fixed serialization $s$ and a fixed lossless compressor $C$,

$$L_C(D) = 8\,|C(s(D))| \quad\text{bits.}$$

This is **not** an information content, and the Note is explicit about it (Proposition 5). It is
simultaneously (i) a computable **upper bound on Kolmogorov complexity**,
$K(D)\le L_C(D)+K(C^{-1})+O(1)$, and (ii) an exact **Shannon codelength** $-\log_2 q_C(s(D))$ under
the crude model $q_C$ that gzip implicitly assumes. Both worlds of Fig. 1c are measured with the
*same* $(s,C)$, so the **ratio** is the claim — never the absolute number.

This resolves the Kolmogorov-vs-Shannon terminology question directly: compressed size is a bound on
the former and an instance of the latter. Earlier drafts called it "gzip codelength"; *codelength*
reads as source code outside information theory, so the paper says **compressed length / compressed
size**.

### How each size is obtained (no payload is ever downloaded)

| Modality | Corpus | Route | Exact? |
| --- | --- | --- | --- |
| Protein | UniProtKB/TrEMBL, Swiss-Prot | HTTP `HEAD` → `Content-Length`; release from `reldate.txt`; entry count from the REST `x-total-results` header | exact |
| DNA / nucleotide | NCBI BLAST `nt` | `nt-nucl-metadata.json` → `bytes-total-compressed`, `number-of-letters` | exact |
| RNA | NCBI BLAST `refseq_rna` | `refseq_rna-nucl-metadata.json` | exact |
| Small molecule | PubChem Compound SDF | one `HEAD` per SDF block (357), rate-limited to NCBI's documented ≤3 req/s | exact |
| Structure | wwPDB divided mmCIF | 1,243 hash-dir listings from `files.wwpdb.org`, sizes to 3 s.f. | ±0.03%/file |

### Measured (2026.07.12, releases in the table)

| Corpus | GB | bits |
| --- | --- | --- |
| UniProtKB/TrEMBL | 40.5 | 3.2 × 10¹¹ |
| UniProtKB/Swiss-Prot | 0.1 | 7.5 × 10⁸ |
| NCBI `nt` | 884.1 | **7.1 × 10¹²** |
| NCBI RefSeq RNA | 67.4 | 5.4 × 10¹¹ |
| PubChem Compound | 115.8 | 9.3 × 10¹¹ |
| wwPDB mmCIF (255,710 structures) | 89.2 | 7.1 × 10¹¹ |
| **Total (all rows)** | **1,197.1** | **9.6 × 10¹² ≈ 10¹³** |
| Total (non-overlapping) | 1,129.6 | 9.0 × 10¹² |

**Cross-check that licenses the measure.** `nt` holds 4.06 × 10¹² nucleotides; a four-letter alphabet
cannot write them in fewer than 8.1 × 10¹² bits, and the archive is *distributed* in 7.1 × 10¹² — the
compressed size recovers **87%** of a counting bound derived without reference to any compressor. Two
unrelated routes agree, so the number tracks sequence content rather than file formatting.

### Traps found the hard way

- **rsync is out.** The wwPDB-documented bulk route (`rsync.rcsb.org::ftp_data`, `--list-only`) gives
  exact sizes in one command, but port 873 is firewalled here (connect timeout), and macOS ships
  `openrsync`. HTTPS gives the same sizes.
- **`files.rcsb.org` returns no `Content-Length`** — it sits behind CloudFront and streams chunked, so
  per-entry `HEAD` is useless. `files.wwpdb.org` *does* return exact `Content-Length`, but 2.6 × 10⁵
  HEADs would be abusive; hence the directory-index route.
- **NCBI 503s under concurrency.** 12 parallel HEADs to PubChem get rejected immediately. Sequential
  with a 0.34 s gap (≤3 req/s) is reliable.
- **PubChem is ~116 GB, not ">300 GB".** The larger figure that circulates counts `CURRENT-Full`
  *including* the XML/ASN representations, not the SDF a chemistry encoder consumes. This is exactly
  why the number had to be measured rather than cited.
- **`2026_02`** — the UniProt release string carries a bare underscore, which is a LaTeX subscript.
  Escaped in the table writer.

### Deliberately excluded

Predicted-structure corpora (AlphaFold DB) would add roughly an order of magnitude, but they are model
*outputs*, not primary observations. Including them would widen the gap while weakening the
"persistent entity" reading, so they are left out and remarked on instead.

### Open

The **contingent-observation** side (~1.4 × 10⁹ compressed bytes → ~10¹⁰ bits) is still computed on
gilahyper and is the only number in Supplementary Note 5 not produced by a committed script. It needs
to move into `experiments/016-information-accounting/scripts/` and regenerate Table S5
(`tab:datasets`) with a compressed-size column — that table currently lists 27 studies (~4.4 × 10⁷
genotypes) against the gilahyper table's 34 (~7.9 × 10⁷ instances).
