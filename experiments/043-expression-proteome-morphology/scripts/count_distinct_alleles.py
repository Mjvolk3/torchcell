# experiments/043-expression-proteome-morphology/scripts/count_distinct_alleles.py
# [[experiments.043-expression-proteome-morphology.scripts.count_distinct_alleles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/count_distinct_alleles
"""Count distinct CDS and protein alleles per reference gene in the Peter 2018 gene-keyed
tarball (allReferenceGenesWithSNPsAndIndelsInferred.tar.gz). CPU only, one streaming pass
(about 3 minutes on GilaHyper).

Writes ``results/distinct_alleles.json`` with per-gene counts, totals and the sorted list
of isolate ids seen in the tarball (so a segregant parent's Peter id can be checked for
membership). The number of distinct alleles bounds the embedding compute for the
isolates: an identical allele shares one embedding row. Proteins are the translation of
the CDS truncated to whole codons (Biopython standard table, stop as ``*``).

    python experiments/043-expression-proteome-morphology/scripts/count_distinct_alleles.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from Bio.Seq import Seq
from dotenv import load_dotenv

from torchcell.utils.paths import experiment_results_dir

EXPERIMENT = "043-expression-proteome-morphology"
TAR_REL = (
    "torchcell-genomes/peter2018_1011_assemblies/"
    "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
)


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """(header, sequence) pairs of a FASTA string."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header = line[1:]
            buf = []
        else:
            buf.append(line.strip())
    if header is not None:
        records.append((header, "".join(buf)))
    return records


def main() -> None:
    """One pass over the tarball; write the counts json."""
    load_dotenv()
    tar = Path(os.environ["DATA_ROOT"]) / TAR_REL
    out = Path(experiment_results_dir(EXPERIMENT, __file__)) / "distinct_alleles.json"
    t0 = time.time()
    per_gene: dict[str, dict[str, int]] = {}
    isolates: set[str] = set()
    n_records = 0
    n_files = 0
    total_cds_bp = 0
    distinct_cds_total = 0
    distinct_prot_total = 0
    with tarfile.open(tar, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".fasta"):
                continue
            gene = Path(member.name).stem
            handle = tf.extractfile(member)
            if handle is None:
                raise RuntimeError(member.name)
            recs = parse_fasta(handle.read().decode())
            n_files += 1
            cds_hash: dict[str, str] = {}
            n_iso = 0
            for header, seq in recs:
                token = header.split()[0]
                # the first header of a file is the reference coordinate line; isolate
                # headers carry <isolate>_<gene>_<symbol>
                if f"_{gene}_" in token:
                    iso = token.split(f"_{gene}_", 1)[0]
                    isolates.add(iso)
                    n_iso += 1
                    n_records += 1
                    total_cds_bp += len(seq)
                    h = hashlib.sha1(seq.upper().encode()).hexdigest()
                    cds_hash.setdefault(h, seq)
            prots = set()
            for seq in cds_hash.values():
                s = seq.upper()
                s = s[: len(s) - len(s) % 3]
                prots.add(str(Seq(s).translate()))
            per_gene[gene] = {
                "isolate_records": n_iso,
                "distinct_cds": len(cds_hash),
                "distinct_protein": len(prots),
            }
            distinct_cds_total += len(cds_hash)
            distinct_prot_total += len(prots)
            if n_files % 500 == 0:
                print(f"{n_files} genes, {time.time() - t0:.0f} s", file=sys.stderr)
    summary = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/count_distinct_alleles.py",
        "tarball": str(tar),
        "gene_files": n_files,
        "isolates_seen": len(isolates),
        "isolate_gene_records": n_records,
        "total_cds_bp": total_cds_bp,
        "distinct_cds_alleles_total": distinct_cds_total,
        "distinct_protein_alleles_total": distinct_prot_total,
        "seconds": round(time.time() - t0, 1),
        "isolates": sorted(isolates),
        "per_gene": per_gene,
    }
    out.write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("per_gene", "isolates")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
