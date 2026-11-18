# experiments/048-pputida-isoprenol-cgt/scripts/provision_string_kt2440.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.provision_string_kt2440]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/provision_string_kt2440
"""Mirror the STRING v12.5 slice for P. putida KT2440, pinned by sha256.

The whole STRING database is three SQL dumps totalling 319.3 GB compressed
(items 39.7, network 226.3, evidence 53.3), which is a PostgreSQL deployment
rather than a file mirror. The per-organism slice for one taxon is 36 MB, and it
carries every scored channel the cell graph reads. So the expansion past yeast
is served by mirroring the slice per taxon, and this script is that mirror for
taxon 160488.

The mirror root is ``--root``, defaulting to ``/bulk/torchcell-string``: /scratch
is at 98% with 150 GB free while /bulk has 20 TB, and a reference mirror that
will grow one directory per organism belongs on /bulk.

Writes ``manifest.json`` (one record per file: source_url, retrieval_command,
sha256, bytes, retrieved_at) and ``SHA256SUMS`` into
``{root}/v12.5/160488/``, in the same shape the genome tier uses. STRING
publishes no checksums, so each sha256 is first-observation and is the anchor a
later re-fetch is compared against.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.graph.string_db import StringAsset, StringTaxonMirror

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")

KT2440_TAXON = 160488
DEFAULT_STRING_ROOT = "/bulk/torchcell-string"

# Everything STRING publishes for one organism. The scored channels are what the
# cell graph reads; the rest is mirrored in the same pass because a second visit
# to a source is a second chance for it to have changed, and because the
# clusters, orthology and enrichment tables are what a later cross-organism
# transfer study needs.
ASSETS = (
    StringAsset.LINKS,
    StringAsset.LINKS_DETAILED,
    StringAsset.LINKS_FULL,
    StringAsset.PHYSICAL_LINKS,
    StringAsset.PHYSICAL_LINKS_DETAILED,
    StringAsset.PHYSICAL_LINKS_FULL,
    StringAsset.REGULATORY_LINKS,
    StringAsset.REGULATORY_LINKS_DETAILED,
    StringAsset.REGULATORY_LINKS_FULL,
    StringAsset.REGULATORY_EVENTS,
    StringAsset.INFO,
    StringAsset.ALIASES,
    StringAsset.SEQUENCES,
    StringAsset.HOMOLOGY,
    StringAsset.ENRICHMENT_TERMS,
    StringAsset.ORTHOLOGY,
    StringAsset.CLUSTERS_PROTEINS,
    StringAsset.CLUSTERS_INFO,
    StringAsset.CLUSTERS_TREE,
)


def main() -> None:
    """Mirror the taxon slice and report what landed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_STRING_ROOT)
    parser.add_argument("--taxon", type=int, default=KT2440_TAXON)
    parser.add_argument("--release", default="12.5")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    mirror = StringTaxonMirror(root=args.root, taxon=args.taxon, release=args.release)
    manifest = mirror.fetch(assets=ASSETS, organism="Pseudomonas putida KT2440")

    verification = mirror.verify()
    bad = [name for name, ok in verification.items() if not ok]
    total = sum(record.bytes for record in manifest.files)

    report = {
        "taxon": manifest.taxon,
        "release": manifest.release,
        "dir": mirror.dir,
        "n_files": len(manifest.files),
        "total_bytes": total,
        "verified": len(verification) - len(bad),
        "mismatched": bad,
        "files": [
            {
                "path": record.path,
                "asset": record.asset,
                "role": record.role,
                "bytes": record.bytes,
                "sha256": record.sha256,
            }
            for record in manifest.files
        ],
    }
    out = osp.join(RESULTS, f"string_mirror_{manifest.taxon}_v{manifest.release}.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"mirrored {len(manifest.files)} files, {total / 1e6:.1f} MB -> {mirror.dir}")
    for record in manifest.files:
        print(f"  {record.bytes:>10,}  {record.sha256[:12]}  {record.path}")
    if bad:
        raise SystemExit(f"sha256 mismatch after write: {bad}")
    print(f"all {len(verification)} files verified against the manifest")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
