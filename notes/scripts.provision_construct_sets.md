---
id: dum928a8bvhdq460oojz0fx
title: Provision_construct_sets
desc: ''
updated: 1792054860981
created: 1792054860981
---

## 2026.10.15 - Depositing the Cachera 2023 construct set

`scripts/provision_construct_sets.py` is the construct-store counterpart of
`scripts/provision_yeast_genomes.py`. It deposits no bytes: the 17 GenBank maps already
sit in the library tier under
`torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/plasmids/`, each pinned in
that key's `manifest.json`, and what is written is the index at
`$DATA_ROOT/torchcell-constructs/cachera2023_cri_spa_plasmids/manifest.json`.

Run of 2026-10-15 (`PYTHONPATH=$WT python scripts/provision_construct_sets.py`):

```text
(a) 17 maps listed in library/cacheraCRISPAHighthroughputMethod2023
(b) chain verified: all 17 maps reproduce from si/gkad656_supplemental_files.zip
    -> plasmid sequences.zip (sha256 59c1c5c2d7cf)
(c) parsed: 6 contigs, 11 blocked
(d) index: 17 files, 6 contigs, 11 blocked, provenance_complete=True
(e) deposited .../cachera2023_cri_spa_plasmids/manifest.json; verified 6 contigs
```

### How `provenance_complete: True` was earned

Review 2 finding 12 says the claim was not earned, and it was right about the KEY
manifest: all 17 files and the container zip have `retrieval: null` there, and
`si_data_sources` is `[]`. The construct index earns it per file instead:

- every one of the 17 files carries its own `RetrievalRecord` (`manual_browser`,
  `source_url`, the recipe as `retrieval_command`, `sha256`, `retrieved_at`), with every
  field READ at runtime from `si/gkad656_supplemental_files.provenance.json`, which is
  itself listed and sha256-pinned in the key manifest and pinned again in each record's
  params. Nothing is typed into the script that is not read from a hashed artifact;
- the chain is **verified, not asserted**: step (b) re-extracts `plasmid sequences.zip`
  from the pinned container in memory and matches all 17 member sha256s. 17 of 17
  reproduce, so a rebuild reads the deposited container and gets these exact bytes;
- `manual_browser` is a complete record by the repo's own definition
  (`RetrievalMethod`): the OUP silverchair URL is CloudFront signed and a bare `curl`
  returns HTTP 403 `MissingKey`, so the deposited bytes are what every rebuild reads and
  re-checking the source is not scriptable. No URL this script cannot fetch is presented
  as one it can.

Three claims that are deliberately NOT conflated: `provenance_complete` is about
retrieval only, each contig's `source_name` is about whether the NAME is sourced, and
`blocked` is about whether a map parsed.

### 6 contigs, 11 blocked

```text
bbaca60509e0  'pBTX1'   LOCUS 'ANTE113_with_NatMX'                      15098 bp  64 features
d823a3fcd08c  'pHO24'   LOCUS 'pHO24_(SNR52p-gRNA(ADE2)-SUP4t_natMX)'    5289 bp  17 features
be6339261ee1  'pHO22'   LOCUS 'pHO22_(gRNA(ChrXV_dwADE2)-NatMX)'         6564 bp  44 features
1c96a0052380  'pHO25'   LOCUS 'pHO25_(SNR52p-gRNA(dw_XII-5)-SUP4t...)'   5289 bp  17 features
df6c198a160e  'pHO29'   LOCUS 'pHO29_-_pCfB3050-hphNT1_(SNR52p-...)'     5843 bp  31 features
2691fa7e905b  'pHO8'    LOCUS 'pHO8_(pCfB257(X-3-LoxP-KlLEU2)-Cas9(Hs))'11436 bp  35 features
```

Blocked, each by exactly one origin-wrapping location the hardened parser refuses:
`pBTX002.gb`, `pCCM023.gb`, `pbtx013.gb`, `pbtx014.gb`, `pbtx016.gb`, `pbtx017.gb`,
`pbtx019.gb`, `pbtx020.gb`, `pbtx021.gb`, `pbtx022.gb`, `pbtx023.gb`. Ten are an `AmpR`
CDS written as `4892..858`; `pBTX002.gb`'s is the `pUC\ori` written as `11559..128`.

**The paper's pBTX2 is among the blocked maps**, so the 8,104 bp XII-5 insert computed in
[[torchcell.sequence.plasmid-and-genomic-content-design]] is NOT indexable yet. That is
the honest cost of refusing the fallback: the previous parser stored that `pUC\ori` as
`[0, 12096)`, the whole plasmid. Unblocking the 11 needs a span that holds parts, which
is scheduled work, not a parser flag.

Note the LOCUS column above: `pBTX001(ante113).gb` carries `LOCUS ANTE113_with_NatMX`,
and `pBTX002.gb` (blocked) carries `LOCUS pBTX001`. A LOCUS-minted contig id would have
named the wrong replicon, which is why identity is the sha256 and each name is quoted
from `paper.md` instead (the script refuses a contig whose quote is not a verbatim
substring of the OCR).

### JBEI: measured, and manual-once

The JBEI public registry is NOT fetched by this script and is not scriptable
anonymously. Measured by reviewer 2: `https://public-registry.jbei.org/` answers 200 (an
HTML app) while `/rest/parts/JBx_266188` and `/rest/search?q=JBx_266188` answer **403**.
Hypothesis (untested): ICE's REST API accepts a session token or API key after login,
which would make retrieval scriptable with credentials; otherwise those handles
(Carruthers 2025, Banerjee 2024, Foo 2014, de Siqueira 2025, Kang 2026) are manual-once
through the web UI with a recorded recipe, exactly as this OUP archive was. Either way
the sequence a registry serves is the DESIGN, not Carruthers' "verified by whole plasmid
sequencing" bytes, which is what `ConstructManifest.sequence_status` exists to say; this
set is `design`.

## 2026.10.15 - 17 of 17, and nine names from SUPPLEMENTARY TABLE S2

Lane T. With a span holding parts the deposit now indexes **17 contigs and blocks 0**
(was 6 and 11). Two things in this script had to change with it.

**The blocked reason comes off the exception.** The `except GenBankLocationError` branch
used to write `BlockedReason.origin_wrapping_location` whatever the refusal was, which is
how all 11 Cachera refusals came to be recorded under one member. It now writes
`refusal.refusal`, the typed member the parser raised with.

**Nine names needed a second artifact.** `build_manifest` refuses to index a map the spec
quotes no name for, and the paper body names only pBTX2 and pCCM023 of the 11. The nine
`pBTX0xx` gene-targeting vectors are named in SUPPLEMENTARY TABLE S2 of
`Supplementary Material.docx`, which is a member of `si/gkad656_supplemental_files.zip`
-- already sha256-pinned in the key manifest and already re-read by step (b). So the
spec gained `si_name_quote_by_path` + `si_member`, and `_container_member_text` extracts
the member in memory, hashes it, and reads `word/document.xml` out as one newline per
paragraph (a table cell is a paragraph) with every tag removed and nothing else changed.
Each quote is checked to be a verbatim substring of that text or the run halts, so the
claim is verified at deposit time rather than asserted, and the member's own sha256 goes
into the quote's `Provenance` with `method="docx-document-xml-paragraph-text"` and
`page="SUPPLEMENTARY TABLE S2: Plasmids Used in this Study"`. Non-breaking spaces are not
normalized; the quotes are written to avoid them.

A path may carry a paper quote or a table quote, never both: one artifact names a
construct.

Re-deposited 2026-10-15; the previous index was moved to
`/scratch/projects/deprecated/2026.10.15/122139-manifest.json` with `trash`, because
`deposit_construct_set` never rewrites an index.
