---
id: 40nodug35lyvsxlsw0u43w6
title: '15'
desc: ''
updated: 1784138744235
created: 1784138744235
---

# Content-addressed interning of constant sub-objects in the LMDB store

## Problem

The component-based `Media`/`Environment` serializes to ~7.9 KB and is stored on
EVERY record (twice: `experiment.environment` + `reference.environment_reference`,
plus the constant `reference`/`publication`). That denormalization ballooned
`dmi_costanzo2016` 45 GB → 159 GB (20.7 M identical copies). General fix: store each
DISTINCT constant sub-object ONCE, content-addressed, and have records carry a tiny
`{"$ref": <hash>}` pointer; reconstruct the full object transparently on read.

## Ground truth (from 3 scouts, file:line)

- **Storage = `pickle.dumps({"experiment": e.model_dump(), "reference": r.model_dump(),
  "publication": p.model_dump()})`**, key = `f"{index}".encode()`; read via
  `pickle.loads` → returns the dict verbatim, NO pydantic re-validation.
  Write block is IDENTICAL in **34** `datasets/scerevisiae/*.py` loaders (e.g.
  `costanzo2016.py:193-200` SMF, `:617-625` DMF, `:976-984` DMI). Base `process()` is
  abstract (`experiment_dataset.py:197-201`).
- **Read choke point** = `ExperimentDataset.get_single_item` (`experiment_dataset.py:274-282`),
  between `pickle.loads` (281) and `return` (282). Covers `get`, the ref-index build
  (`:87`), and `cpu_experiment_loader.py:60`.
- **`reference` is inline per-record** (contributes to blowup). `experiment_reference_index`
  is a SEPARATE `preprocess/*.json` (`:345-380`), rebuilt by reading `dataset[i]["reference"]`
  → resolved reference → hash byte-identical to today ⇒ index unchanged.
- **`@post_process`** (`:118-144`) reads records back immediately (gene_set + ref-index),
  so read-resolve + `max_dbs>=2` on the READ env open (`_init_db` `:210-216`) are
  MANDATORY, not deferrable.
- **`compute_gene_set_sequential`** (`:302-322`) scans the DEFAULT db with a raw cursor and
  reads only `["experiment"]["genotype"]` — so (a) interned MUST be a NAMED sub-db (not
  hash rows in the default db) and (b) `genotype` MUST stay inline (never interned).
- **Hash primitives already exist**: `serialize_for_hashing` (`experiment_dataset.py:58-67`)
  - `compute_sha256_hash` (`data/data.py:123-125`); the ref-index already does
  `compute_sha256_hash(serialize_for_hashing(reference))` (`:88`).
- **Reconstruction** elsewhere is `EXPERIMENT_TYPE_MAP[d["experiment"]["experiment_type"]](**...)`
  (`aggregate.py:198`, `deduplicate.py:86`, `conversion.py:168`) + `transform_item`
  (`:401-414`) — so `experiment_type`/`experiment_reference_type` and the 3 top-level keys
  MUST stay in-band; intern only inner constant VALUES.

## Decisions

1. **Intern targets** (constant / near-constant, ≥512 B canonical): `experiment.environment`,
   the whole top-level `reference`, and `publication`. NOT `genotype`, NOT `phenotype`, NOT
   `experiment_type`. `environment` has ≤2 variants/dataset (temp 26/30); `reference`/`publication`
   ~1. Costanzo 20.7 M × 7.9 KB → a handful of interned rows.
2. **Named sub-db `interned`** in the same `data.mdb` (NOT a separate file): open both the
   WRITE env and the READ env (`_init_db`) with `max_dbs=2`; `env.open_db(b"interned")`.
   Records stay integer-keyed in the DEFAULT db.
3. **Pointer shape**: replace the inline value with `{"$ref": <hash>, "name": <hint>}`
   (`hint` = `environment.media.name`, or `dataset_name` for reference/publication — metadata
   only, for a future by-name sparse read). Interned value = `pickle.dumps(sub_dict)`.
4. **Hash** = `compute_sha256_hash(serialize_for_hashing(pydantic_obj))` (reuse; aligns the
   interned reference key with the ref-index hash). Hash the LIVE pydantic object at write
   (we have both object and dict at the hook).
5. **Scope THIS PR**: base helpers + wire the **4 SGA loaders** (`costanzo2016`, `kuzmin2018`,
   `kuzmin2020`, `baryshnikova2010`) — the balloon. Read-resolve is centralized so the other
   30 loaders' (non-interned) LMDBs read as a no-op; they adopt the write helper incrementally.
6. **NaN**: `serialize_for_hashing` uses `json.dumps` (allow_nan) → stable token; interned
   objects are constant per dataset ⇒ consistent hash. Pickle stores NaN natively. No change.

## Implementation

**`torchcell/data/experiment_dataset.py` (base — the centralization):**

- `INTERN_MIN_BYTES = 512`; `INTERN_DB = b"interned"`.
- `_intern_record(experiment, reference, publication, interned_txn, seen: dict, lock) -> bytes`:
  build `rec = {"experiment": experiment.model_dump(), "reference": reference.model_dump(),
  "publication": publication.model_dump()}`; for each intern target compute
  `h = compute_sha256_hash(serialize_for_hashing(obj))`; if `len(serialize_for_hashing(obj)) >=
  512`: `interned_txn.put(h.encode(), pickle.dumps(sub_dict))` guarded by `seen`+`lock`
  (put-if-absent), replace `rec[...]=  {"$ref": h, "name": hint}`; else leave inline. Return
  `pickle.dumps(rec)`. Handles nested `experiment.environment`.
- `_resolve_interned(obj)` (module fn): recursive walk; a dict `== {"$ref","name"?}` with a
  `$ref` key → `self._interned[obj["$ref"]]` (already a dict); else recurse dict/list; else
  return as-is. No-op when no `$ref` present (backward compat).
- `self._interned: dict[str, dict] | None = None` in `__init__` (~167, survives `close_lmdb`);
  `_load_interned()` populates once from the named db (tolerate absent db → `{}`).
- `get_single_item` (`:274-282`): after `pickle.loads`, `return self._resolve_interned(data)`
  (calls `_load_interned()` first).
- `_init_db` (`:210-216`): add `max_dbs=2`.
- Write-env helper `_open_write_lmdb(path) -> (env, interned_db)` with `max_dbs=2`,
  `map_size=int(1e12)` (unchanged; interning only shrinks).

**4 SGA loaders**: replace the inline `serialized_data = pickle.dumps({...}); txn.put(...)`
with `txn` from the interned-aware env + `self._intern_record(...)`. Each `lmdb.open(...)` in
these 4 files → `self._open_write_lmdb(...)` (get `interned_db`), open a second `write=True`
txn on `interned_db` inside the batch (single-writer serializes; a `threading.Lock`+`seen`
dedups across io_workers threads). SMF sequential, DMF/DMI batched via ThreadPoolExecutor.

## Verification (required in the PR)

- Unit: two records sharing an `Environment` → interned db has 1 environment row; round-trip
  `_intern_record` then `_resolve_interned` reconstructs the exact stored dict.
- Backward-compat: a legacy record dict (no `$ref`) → `_resolve_interned` returns it unchanged.
- Integration: clear `smf_baryshnikova2010/{processed,preprocess}`, run
  `python torchcell/datasets/scerevisiae/baryshnikova2010.py`; assert `ds[0]` returns full
  `environment` with `media.name` containing "SGA" + `is_synthetic=True`; interned db has 1–2
  environments; records-db bytes ≪ prior inline. `@post_process` (gene_set + ref-index)
  completes with no ValidationError.
- `ruff` + `mypy --strict` clean. Do NOT rebuild Costanzo/Kuzmin here.

## Risks / follow-ups (flag, don't fix here)

- **Raw downstream readers** (`dataset_readers/reader.py`, neo4j/BioCypher adapters,
  `data/aggregate.py`/`deduplicate.py` if they `pickle.loads` directly) must call
  `_resolve_interned` before consuming an interned LMDB — matters at the (not-yet-active) graph
  build. Export `_resolve_interned` as a reusable fn.
- The other **30 loaders** keep inline writes (read is a no-op); migrate to `_intern_record`
  later.
- After landing + baryshnikova verify: **full SGA re-rebuild** (separate step) on the lean
  encoding; expect 159 GB → ~45 GB.

## 2026.07.15 - Write-path finished (separate-env) + Baryshnikova verified GREEN

Finished the write-path migration from the (abandoned) named LMDB sub-db to a **separate
sibling `interned` env** at `processed/interned/` — the named sub-db registered its name as
a key in the records env, poisoning `compute_gene_set_sequential`'s raw cursor
(`_pickle.UnpicklingError: invalid load key '\x00'`). The sibling env keeps the records env a
pristine `0..N` keyspace.

- **Code:** `_intern_record(...itxn)` (was `txn, interned_db`); all **14 write sites** across
  `baryshnikova2010` (×1 inline), `costanzo2016` (×1 inline + ×2 `_process_batch`
  ThreadPoolExecutor), `kuzmin2018` (×5 inline), `kuzmin2020` (×5 inline) now open a paired
  `with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:` (records env
  locked first → no cross-env deadlock) and close both. Removed dead `INTERN_DB_NAME`; test
  moved to the sibling-env API.
- **Gates:** ruff clean; mypy --strict clean (5 files); 4 unit tests pass.
- **Baryshnikova rebuild (real LMDB):** 5993 records — **identical count to both
  pre-interning backups** (interning is record-count-neutral). Raw `record[0]`:
  `environment` + `reference` are `{"$ref"}` pointers; `publication` (<512 B) + `genotype`
  stay inline. Interned env = **2 objects** (the one Environment + one Reference, shared by
  all 5993). Public `ds[0]` resolves to full component media (name has "SGA",
  `is_synthetic=True`, 9 components); `gene_set` (5436) builds with no `\x00`. Records env
  keys all-integer.
- **Storage:** environment field **187 B (`$ref`) vs 4632 B resolved → 25×**; whole dataset
  on disk **48 MB → 11 MB (4.4×)**.
- **Deferred (gated on this):** `nq merge`; full `dmi/dmf_costanzo2016` + `dmi_kuzmin*`
  re-rebuilds; ChEBI/InChIKey/SMILES resolver; migrating the other ~30 loaders.
