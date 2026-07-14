# experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py
# [[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes
"""Measure the *persistent-entity* corpora behind Fig. 1c and its Supplementary Note.

Fig. 1c contrasts two worlds:

  persistent entities      DNA, RNA, protein, small molecules, structures -- molecular
                           identities that do not change between assays.
  contingent observations  our phenotype corpus -- measurements conditioned on
                           genotype x environment x assay x strain x time.

This script measures the FIRST world by asking each public archive how large its
distributed, compressed artifact is. It NEVER downloads a payload: sizes come from HTTP
`Content-Length`, from NCBI BLAST metadata JSON, or from the wwPDB directory index. Every
number is therefore re-derivable by anyone with a network connection -- which is the point.
The Supplementary Note cites this script, not our word for it.

THE MEASURE. For a corpus D with a fixed serialization s and compressor C we report

    L_C(D) = 8 * |C(s(D))|   bits            ("compressed size")

This is (i) a computable UPPER BOUND on the Kolmogorov complexity, K(D) <= L_C(D) +
K(decompressor) + O(1), and (ii) exactly a Shannon codelength -log2 q(D) under gzip's
crude implicit model q. It is NOT an intrinsic information content -- it depends on the
serialization and the compressor. It is a like-for-like yardstick applied to both worlds,
so the RATIO is what we claim, to order of magnitude.

CROSS-CHECK. Where an archive reports a raw letter count we also compute the naive
alphabet floor (2 bits/base for nucleotide). For NCBI `nt` the two independent routes agree
within ~15%, which is what licenses reading the compressed size as a scale measure rather
than a compression artifact.

KNOWN OVERLAP. `refseq_rna` transcripts are also present in `nt`. The headline total is
therefore reported twice: summing all rows, and summing a non-overlapping subset
(nt + TrEMBL + PubChem + PDB). Both land at ~1e13 bits, so the claim is insensitive to it.

NOT INCLUDED, deliberately: predicted-structure corpora (AlphaFold DB, ~2e5 GB) would add
an order of magnitude but are model *outputs*, not primary observations -- including them
would only widen the gap while weakening the "persistent entity" reading.

PROVENANCE -- read this before citing a number. These archives are LIVE and grow monotonically,
so the script is reproducible in *method* but not in *value*: re-run it next month and the entity
total is larger. Every measurement therefore writes an immutable, timestamped snapshot that the
paper cites, and each row records the archive's own release id (UniProt `2026_02`, `nt`'s
`last-updated`, ...) alongside `fetched_utc`. If a referee re-runs this and gets bigger numbers,
that is the expected behaviour and it only widens the gap in Fig. 1c -- the published comparison is
conservative. The two rows that are a SUM rather than a single authoritative field (PubChem, wwPDB)
additionally dump every per-file size to an audit file, so the arithmetic can be re-checked without
re-querying anything.

Outputs
  results/persistent_entity_corpus_sizes.csv                          latest (overwritten)
  results/persistent_entity_corpus_sizes.json                         latest (overwritten)
  results/snapshots/persistent_entity_corpus_sizes_<ts>.csv           IMMUTABLE -- cite this
  results/snapshots/persistent_entity_corpus_sizes_<ts>.json          IMMUTABLE -- URLs + methods
  results/snapshots/persistent_entity_corpus_sizes_<ts>.audit.csv.gz  IMMUTABLE -- every per-file size
  paper/nature-biotech/sections/tab-entity-corpora.tex                (--write-table)

Runtime ~4 min: PubChem needs one HEAD per SDF block, rate-limited to NCBI's documented
<=3 req/s; wwPDB needs 1,243 directory listings. rsync (the wwPDB-documented bulk route) is
NOT used -- port 873 is firewalled on many networks, and HTTPS gives the same sizes.

Run from the repo root:
  python experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py --write-table
"""
import argparse
import json
import os
import os.path as osp
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from torchcell.timestamp import timestamp

UA = {"User-Agent": "torchcell-information-accounting (github.com/Mjvolk3/torchcell)"}

UNIPROT = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/"
BLAST = "https://ftp.ncbi.nlm.nih.gov/blast/db/"
PUBCHEM = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"
WWPDB = "https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/"

OUT_DIR = osp.join("experiments", "016-information-accounting", "results")
SNAP_DIR = osp.join(OUT_DIR, "snapshots")
TEX_OUT = osp.join("paper", "nature-biotech", "sections", "tab-entity-corpora.tex")

# nt and refseq_rna overlap (RefSeq transcripts are in nt); drop refseq_rna from the
# conservative total rather than double-counting it.
NON_OVERLAPPING = {"UniProtKB/TrEMBL", "NCBI nt", "PubChem Compound", "wwPDB mmCIF"}


def _open(url, method="GET", attempts=4):
    """Fetch with a bounded retry on TRANSIENT faults only.

    This is not a fallback and it masks nothing: every attempt hits the same authoritative
    endpoint, a 4xx re-raises immediately (a 404 is a real error, not a hiccup), and after the
    last attempt the exception propagates. It exists because a ~5-minute run spanning four hosts
    and ~1,600 requests will occasionally take a DNS or 5xx fault, and a script whose whole point
    is a *citable* measurement must not lose a completed run to one dropped packet.
    """
    req = urllib.request.Request(url, headers=UA, method=method)
    for i in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == attempts - 1:
                raise
            time.sleep(2**i)
        except urllib.error.URLError:  # DNS failure, connection reset, timeout
            if i == attempts - 1:
                raise
            time.sleep(2**i)


def _text(url):
    with _open(url) as r:
        return r.read().decode("utf-8", "replace")


def _json(url):
    with _open(url) as r:
        return json.loads(r.read())


def _content_length(url):
    with _open(url, method="HEAD") as r:
        return int(r.headers["Content-Length"])


def uniprot(corpus, filename, reviewed):
    """UniProtKB FASTA. Exact Content-Length; release from reldate.txt; count from REST."""
    n_bytes = _content_length(UNIPROT + filename)
    release = re.search(r"Release (\S+)", _text(UNIPROT + "reldate.txt")).group(1)
    with _open(
        f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:{reviewed}&size=0",
        method="HEAD",
    ) as r:
        n_items = int(r.headers["x-total-results"])
    return dict(
        modality="Protein",
        corpus=corpus,
        artifact=filename,
        url=UNIPROT + filename,
        release=release,
        n_items=n_items,
        item_unit="sequences",
        letters=None,
        alphabet_bits=None,
        compressed_bytes=n_bytes,
        method="HTTP HEAD Content-Length",
        exact=True,
    )


def blast_db(modality, corpus, dbname):
    """NCBI BLAST database. Metadata JSON carries exact compressed bytes and letter count."""
    url = f"{BLAST}{dbname}-nucl-metadata.json"
    m = _json(url)
    return dict(
        modality=modality,
        corpus=corpus,
        artifact=f"{dbname} ({m['number-of-volumes']} volumes, .tar.gz)",
        url=url,
        release=m["last-updated"][:10],
        n_items=m["number-of-sequences"],
        item_unit="sequences",
        letters=m["number-of-letters"],
        alphabet_bits=2,  # 4-letter nucleotide alphabet
        compressed_bytes=m["bytes-total-compressed"],
        method="BLAST metadata JSON: bytes-total-compressed",
        exact=True,
    )


def pubchem():
    """PubChem Compound SDF. Exact sum of one HEAD per SDF block, rate-limited to <=3 req/s."""
    files = re.findall(r'href="(Compound_[0-9_]+\.sdf\.gz)"', _text(PUBCHEM))
    detail, total = [], 0
    for i, f in enumerate(files, 1):
        n_bytes = _content_length(PUBCHEM + f)
        detail.append((f, n_bytes))
        total += n_bytes
        time.sleep(0.34)  # NCBI allows <=3 requests/s without an API key
        if i % 50 == 0 or i == len(files):
            print(f"    pubchem {i}/{len(files)} blocks, {total / 1e9:.1f} GB", flush=True)
    cid_max = int(re.search(r"_(\d+)\.sdf\.gz$", files[-1]).group(1))
    return dict(
        _detail=detail,
        modality="Small molecule",
        corpus="PubChem Compound",
        artifact=f"{len(files)} SDF blocks, CID 1-{cid_max:,}",
        url=PUBCHEM,
        release=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n_items=len(files),
        item_unit=f"SDF blocks, CID 1--{cid_max:,}",
        letters=None,
        alphabet_bits=None,
        compressed_bytes=total,
        method=f"sum of {len(files)} HTTP HEAD Content-Length",
        exact=True,
    )


def wwpdb():
    """wwPDB experimental structures. Sum every .cif.gz across the divided mmCIF tree.

    files.rcsb.org sits behind CloudFront and returns no Content-Length, so per-file HEAD is
    unavailable; and 2.6e5 HEADs would be abusive anyway. The wwPDB directory index reports
    sizes to 3 significant figures. Rounding is independent per file, so the aggregate error
    over ~2.6e5 files is far below the order-of-magnitude precision we claim.
    """
    unit = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}
    row = re.compile(
        r'href="([^"]+\.cif\.gz)">.*?text-end">\s*([\d.]+)\s*(B|KB|MB|GB)\s*<', re.S
    )
    hash_dirs = sorted(set(re.findall(r'href="([0-9a-z]{2})/"', _text(WWPDB))))

    def scan(h):
        return [(n, int(float(v) * unit[u])) for n, v, u in row.findall(_text(f"{WWPDB}{h}/"))]

    with ThreadPoolExecutor(max_workers=6) as ex:  # 6, not 8: 8 bursts the local DNS resolver
        sizes = [f for chunk in ex.map(scan, hash_dirs) for f in chunk]
    print(f"    wwpdb {len(hash_dirs)} hash dirs, {len(sizes):,} .cif.gz files", flush=True)
    return dict(
        _detail=sizes,
        modality="Structure",
        corpus="wwPDB mmCIF",
        artifact="divided/mmCIF coordinate files (.cif.gz)",
        url=WWPDB,
        release=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n_items=len(sizes),
        item_unit="structures",
        letters=None,
        alphabet_bits=None,
        compressed_bytes=sum(b for _, b in sizes),
        method=f"sum of {len(hash_dirs)} wwPDB directory-index listings (sizes to 3 s.f.)",
        exact=False,
    )


def measure():
    print("[1/5] UniProtKB/TrEMBL (protein, unreviewed)", flush=True)
    trembl = uniprot("UniProtKB/TrEMBL", "uniprot_trembl.fasta.gz", "false")
    print("[2/5] UniProtKB/Swiss-Prot (protein, reviewed)", flush=True)
    sprot = uniprot("UniProtKB/Swiss-Prot", "uniprot_sprot.fasta.gz", "true")
    print("[3/5] NCBI nt + refseq_rna (nucleotide)", flush=True)
    nt = blast_db("DNA / nucleotide", "NCBI nt", "nt")
    rna = blast_db("RNA (transcripts)", "NCBI RefSeq RNA", "refseq_rna")
    print("[4/5] PubChem Compound SDF (small molecule) -- ~3 min, rate-limited", flush=True)
    chem = pubchem()
    print("[5/5] wwPDB divided mmCIF (structure) -- ~1 min", flush=True)
    pdb = wwpdb()

    rows = [trembl, sprot, nt, rna, chem, pdb]
    # PubChem and wwPDB are the only rows that are a SUM rather than a single authoritative field,
    # so their per-file sizes are kept: the arithmetic must be re-checkable without re-querying.
    audit = pd.DataFrame(
        {"corpus": r["corpus"], "file": name, "compressed_bytes": n_bytes}
        for r in rows
        for name, n_bytes in r.pop("_detail", [])
    )

    df = pd.DataFrame(rows)
    df["compressed_bits"] = df["compressed_bytes"] * 8
    df["alphabet_floor_bits"] = df["letters"] * df["alphabet_bits"]
    df["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df["snapshot_id"] = timestamp()
    return df, audit


def _sci(x, sig=2):
    """1.23e+11 -> $1.2\\times10^{11}$"""
    s = f"{x:.{sig - 1}e}"
    mant, exp = s.split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def _tex(s):
    """UniProt releases look like `2026_02`; a bare underscore is a LaTeX subscript."""
    return str(s).replace("_", r"\_")


def _contents(r):
    """What the entry count actually counts -- the units differ per archive, so say so."""
    n = f"{int(r.n_items):,} {r.item_unit}"
    if pd.notna(r.letters):
        n += f" ({_sci(r.letters)} nt)"
    return n


def latex_table(df, total_all, total_cons):
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r.modality} & {r.corpus} & {_tex(r.release)} & {_contents(r)} & "
            f"{r.compressed_bytes / 1e9:,.1f} & {_sci(r.compressed_bits)}\\\\"
        )
    body = "\n".join(rows)
    snap = df.snapshot_id.iloc[0]
    fetched = df.fetched_utc.iloc[0]
    return rf"""%% SOURCE: experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py
%% AUTO-GENERATED -- do not hand-edit; edits are lost on the next run.
%% Regenerate: python experiments/016-information-accounting/scripts/persistent_entity_corpus_sizes.py --write-table
%% Measured {fetched}. The immutable snapshot behind every cell below -- each URL queried, each
%% archive release id, and every per-file size summed for the PubChem and wwPDB rows -- is
%%   experiments/016-information-accounting/results/snapshots/persistent_entity_corpus_sizes_{snap}.{{csv,json}}
%%   experiments/016-information-accounting/results/snapshots/persistent_entity_corpus_sizes_{snap}.audit.csv.gz
%% Cite that snapshot if these numbers are questioned; the archives are live and will have grown.
\begin{{table*}}[t]
\centering
\footnotesize
\caption{{\textbf{{Persistent-entity corpora.}} Compressed size of each public archive in the
representation it is actually distributed in, obtained without downloading any payload
(HTTP \texttt{{Content-Length}}, NCBI BLAST metadata, or the wwPDB directory index). Compressed
size $L_C(D)=8\lvert C(s(D))\rvert$ is a computable upper bound on Kolmogorov complexity and a
codelength under \texttt{{gzip}}'s implicit model, not an intrinsic information content; it is
applied identically to both worlds of Fig.~\ref{{fig:torchcell}}c, so the \emph{{ratio}} is what is
claimed. RefSeq transcripts also occur in \texttt{{nt}}, so the conservative total omits them.
Sizes are decimal GB ($10^9$\,bytes); PubChem and the PDB are rolling archives, so their
\emph{{Release}} is the fetch date. Measured {fetched[:10]} ({fetched[11:19]}~UTC) by the script named
above and archived as a dated snapshot; no number here is hand-entered. These archives grow
monotonically, so re-running the script returns a \emph{{larger}} entity total and a larger ratio---the
separation reported here is conservative.}}
\label{{tab:entity-corpora}}
\begin{{tabular}}{{@{{}}l l l l r r@{{}}}}
\toprule
\textbf{{Modality}} & \textbf{{Corpus}} & \textbf{{Release}} & \textbf{{Contents}} & \textbf{{GB}} & \textbf{{Bits}}\\
\midrule
{body}
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Total}} (all rows)}} & {total_all / 1e9:,.1f} & {_sci(total_all * 8)}\\
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Total}} (non-overlapping: \texttt{{nt}} $+$ TrEMBL $+$ PubChem $+$ PDB)}} & {total_cons / 1e9:,.1f} & {_sci(total_cons * 8)}\\
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--write-table",
        action="store_true",
        help=f"also write the Supplementary table to {TEX_OUT}",
    )
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="rebuild the outputs from the saved CSV instead of re-querying the archives "
        "(use when only the table formatting changed -- do not re-hammer NCBI)",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = osp.join(OUT_DIR, "persistent_entity_corpus_sizes.csv")
    json_path = osp.join(OUT_DIR, "persistent_entity_corpus_sizes.json")

    # --from-csv is a re-format, not a measurement: it must not mint a new snapshot, and it must
    # keep the snapshot_id of the run whose numbers it is re-rendering.
    df, audit = (pd.read_csv(csv_path), None) if args.from_csv else measure()
    total_all = int(df.compressed_bytes.sum())
    total_cons = int(df[df.corpus.isin(NON_OVERLAPPING)].compressed_bytes.sum())
    snap = df.snapshot_id.iloc[0]

    record = {
        "measure": "L_C(D) = 8*|C(s(D))| bits; gzip on the distributed serialization",
        "caveat": "compressed size upper-bounds Kolmogorov complexity and is a codelength under "
        "gzip's implicit model; it is NOT an information content. Only the ratio is claimed.",
        "archives_are_live": "these corpora grow monotonically; a later re-run returns a LARGER "
        "entity total, so the reported separation is conservative. Cite the dated snapshot.",
        "snapshot_id": snap,
        "fetched_utc": df.fetched_utc.iloc[0],
        "rows": json.loads(df.to_json(orient="records")),
        "total_all_bytes": total_all,
        "total_all_bits": total_all * 8,
        "total_non_overlapping_bytes": total_cons,
        "total_non_overlapping_bits": total_cons * 8,
    }

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    if audit is not None:
        os.makedirs(SNAP_DIR, exist_ok=True)
        stem = osp.join(SNAP_DIR, f"persistent_entity_corpus_sizes_{snap}")
        df.to_csv(f"{stem}.csv", index=False)
        with open(f"{stem}.json", "w") as f:
            json.dump(record, f, indent=2)
        audit.to_csv(f"{stem}.audit.csv.gz", index=False, compression="gzip")
        print(f"\nsnapshot {snap}: {len(audit):,} per-file sizes archived (PubChem + wwPDB)")

    print("\n" + "=" * 84)
    print(f"{'modality':<20}{'corpus':<24}{'GB':>10}{'bits':>14}{'floor(2b/base)':>16}")
    print("-" * 84)
    for _, r in df.iterrows():
        floor = f"{r.alphabet_floor_bits:.3g}" if pd.notna(r.alphabet_floor_bits) else "--"
        print(
            f"{r.modality:<20}{r.corpus:<24}{r.compressed_bytes / 1e9:>10,.1f}"
            f"{r.compressed_bits:>14.3g}{floor:>16}"
        )
    print("-" * 84)
    print(f"{'TOTAL (all rows)':<44}{total_all / 1e9:>10,.1f}{total_all * 8:>14.3g}")
    print(f"{'TOTAL (non-overlapping)':<44}{total_cons / 1e9:>10,.1f}{total_cons * 8:>14.3g}")
    print("=" * 84)

    # Cross-check: does gzip on a 2-bit-packed nucleotide archive recover the alphabet floor?
    nt = df[df.corpus == "NCBI nt"].iloc[0]
    print(
        f"\ncross-check  NCBI nt: compressed {nt.compressed_bits:.3g} bits vs "
        f"alphabet floor {nt.alphabet_floor_bits:.3g} bits "
        f"(ratio {nt.compressed_bits / nt.alphabet_floor_bits:.2f})"
    )
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    if audit is not None:
        stem = osp.join(SNAP_DIR, f"persistent_entity_corpus_sizes_{snap}")
        print(f"wrote {stem}.csv\nwrote {stem}.json\nwrote {stem}.audit.csv.gz")
    print(f"\nCITE THIS SNAPSHOT if the numbers are questioned: {snap} ({df.fetched_utc.iloc[0]})")

    if args.write_table:
        with open(TEX_OUT, "w") as f:
            f.write(latex_table(df, total_all, total_cons))
        print(f"wrote {TEX_OUT}")


if __name__ == "__main__":
    main()
