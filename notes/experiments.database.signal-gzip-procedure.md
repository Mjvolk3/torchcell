---
id: 1gex5i67gd6p935q6ms07ot
title: Signal Gzip Procedure
desc: ''
updated: 1783976134446
created: 1783976134446
---

## 2026.07.13 - How the gzip "Signal" is computed (procedure + explainer)

Documents the `Signal (gzip, bytes)` column in
[[paper.supported-datasets-and-databases]]. Code:
`torchcell/paper/tables.py::stream_gzip_signal` (extractor `instance_bytes`),
driven by `experiments/database/scripts/build_supported_datasets_table.py`;
per-dataset CLI `python -m torchcell.paper.signal <subpath>`.

### One sentence

We turn every record in a dataset into a string of bytes, glue all of them into
one long stream, gzip that single stream once, and report the size of the gzip
output in bytes. That size is the "Signal". Computing it requires a **full
sequential read of every record** in the dataset -- there is no shortcut, gzip
must see the actual bytes (this is why 30M-record / 100+ GB datasets take tens of
minutes).

### Step 1 -- serialization = turning a Python object into bytes

Each LMDB record is a Python `dict` (the `experiment`: `genotype`/perturbations,
`environment`, `phenotype`). Compressors operate on **bytes**, not live Python
objects, so we must first flatten the dict into a linear byte sequence that fully
represents it. "Bytes" = a sequence of numbers 0-255; the text `{"fitness": 0.9}`
becomes the byte values of those characters.

- **Pickle** is Python's native serialization; it is how the record is already
  stored on disk. `pickle.loads(value)` deserializes it back into the dict.
- We do **not** gzip the pickle. We pull out the `experiment` sub-dict and
  **re-serialize it as JSON with sorted keys**:
  `json.dumps(experiment, sort_keys=True, default=str).encode()`.
  - `sort_keys=True` -> canonical & deterministic. **Why this is necessary:** a
    Python dict has no guaranteed key order, so the *same content* with keys in a
    *different order* produces different bytes and a slightly different compressed
    total (approximately the same, not exact -- gzip is mildly order-sensitive).
    Sorting fixes one canonical ordering so identical content always yields the
    *bit-exact same* Signal, reproducible and comparable across datasets/runs.
  - A "byte" is 8 bits -- eight 0/1s. Text is stored as one (ASCII) or more (UTF-8)
    bytes per character, so `.encode()` really does turn the string into a run of
    binary numbers 0-255, which is what the compressor reads.
  - `default=str` -> coerce non-JSON types (enums, etc.) to strings so they serialize.
  - `.encode()` -> turn the JSON *text* into UTF-8 *bytes*. That is "serialize to bytes".
  - We re-serialize (rather than reuse the pickle or the raw `.mdb`) so we compress
    the **content only** -- which genes, which conditions, which values -- not LMDB's
    storage layout or pickle's Python-object framing.

### Step 2 -- gzip / DEFLATE / level 6 / wbits (the jargon)

- **gzip** is a file format; **DEFLATE** is the algorithm inside it (same one in
  `.zip`/PNG). DEFLATE (a) finds repeated byte runs and replaces later copies with a
  short "go back N bytes, copy M" reference (LZ77), then (b) Huffman-codes the
  result. Repetition -> shrinkage.
- **level 6** = the speed/size dial (1 = fast/big, 9 = slow/small, 6 = the standard
  default). Using the default makes our number match a plain command-line `gzip`.
- **`zlib.DEFLATED`** = "use the DEFLATE method" (a required argument; the only real option).
- **`16 + zlib.MAX_WBITS`** = `MAX_WBITS` (15) uses the largest 32 KB look-back window
  (best compression); `+16` = "wrap the output in a gzip container" (small header +
  trailing checksum) instead of raw headerless DEFLATE. So it means "produce a standard
  gzip stream with the biggest window."

### Step 3 -- streaming: `compress()`, `flush()`, and the "remainder"

We never build the giant glued-together blob in memory (tens of GB for Costanzo).
Instead we make **one** compressor and feed it each record's bytes in turn:

- A compressor is **stateful**: feeding it a chunk emits *some* output but keeps a
  buffer internally (it may still find a repeat in the next chunk), so each
  `compress()` returns a partial (sometimes empty) slice.
- At the end, `flush()` says "no more data -- finalize": it emits whatever is still
  buffered plus the gzip trailer (checksum + length). **That trailing emission is the
  "remainder."**
- We add up only the **lengths** of what `compress()` and `flush()` emit. That sum is
  the total gzip size = the Signal.

### Step 4 -- it is ONE zip, not per-instance zipping

Feeding chunks to a single compressor produces the **identical compressed size** as
concatenating every record's bytes into one blob and calling `gzip` once (asserted by
a unit test, `test_stream_gzip_signal_matches_nonstreaming`). Streaming is purely a
memory trick; instances are never zipped separately. (Precisely: the *size* is
identical; only the gzip header's mtime field could differ, and size is all we report.)

### Step 5 -- cross-record redundancy and order

Because it is one continuous stream, when a later record repeats an earlier gene name
or boilerplate string, DEFLATE emits a tiny back-reference instead of storing it again.
That is why Caudal's 5,014 highly-repetitive perturbations (~2.6 MB raw/record)
contribute far less than their raw size -- the **diversity**, not the repetition, costs
bytes.

- **Order matters, but only slightly:** gzip is order-sensitive -- adjacent
  similar records compress a little better -- so a different record order gives a
  *slightly* different byte total. It is **approximately the same, not exact**: a
  shuffle perturbs the number by a small amount, it does not change its magnitude.
  We nonetheless fix the order (LMDB iterates in **sorted key order**) so the
  Signal is *bit-exactly reproducible*, not merely close, run to run. So: the value
  is well-defined and deterministic, though not strictly order-invariant.

### What gzip does NOT know (why it is a bound, not an entropy)

A natural worry: "gzip can only compress if it knows the data's distribution."
It does **not** know the true distribution. gzip carries a **fixed, generic model**
(LZ77 back-references + Huffman coding); it exploits only the redundancy that model
can see, and nothing about the biology or the true source. So the compressed size
is a **computable upper bound on Kolmogorov complexity** and an exact codelength
under gzip's *implicit* model -- **not** the entropy of the source. This is exactly
Proposition 5 of Supplementary Note 5 (`L_C(D)`): a bound and a codelength, not an
information content, so only the **ratio** between corpora (measured with the same
serialization + compressor) is claimed, to order of magnitude. A learned model
(e.g. a protein language model over TrEMBL) would compress far better than gzip --
the gap is precisely the evolutionary structure gzip cannot see.

### Step 6 -- return value and meaning

`stream_gzip_signal` returns `(n, total)`: `n` = record count (the **Instances**
column), `total` = total gzip bytes (the **Signal (gzip, bytes)** column).

Compressed size approximates the dataset's **non-redundant information** -- a practical,
comparable proxy for Kolmogorov complexity (the length of the shortest program that
reproduces the data). Boilerplate/repetition shrink toward zero; genuine diversity
stays large. That is why a 1,484-strain expression set (6,169-dim vectors) outscores a
20M-row scalar-fitness set: the fat vectors carry more distinct structure than millions
of near-duplicate scalars. It also relocates fitness/interaction signal onto the
**combinatorial genotype** (which gene pairs/triples), since the perturbation is now
part of the compressed instance.

### Reproducible recipe (one dataset)

```python
import lmdb, pickle, json, zlib
env = lmdb.open("<root>/processed/lmdb", readonly=True, lock=False)
comp = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)  # gzip, level 6, 32KB window
signal = n = 0
with env.begin() as txn:
    for _, v in txn.cursor():                                   # full sequential scan
        inst = pickle.loads(v)["experiment"]                    # perturbation + environment + phenotype
        blob = json.dumps(inst, sort_keys=True, default=str).encode()
        signal += len(comp.compress(blob)); n += 1
signal += len(comp.flush())                                     # the "remainder" + gzip trailer
# signal = Signal (gzip, bytes); n = Instances
```

### What is NOT counted

The raw `.mdb` storage overhead; the `reference`/`publication` sub-records (only
`experiment` is compressed); the shared reference genome and any external sequence
payload (natural-isolate perturbations hold only `sequence_uri` + `sequence_sha256`
pointers, not nucleotides). So the number measures the information in the **stored
instance representation** (including stored annotation: SE, `n_replicates`,
`measurement_type`, ontology IDs, descriptions), not a minimal encoding.
