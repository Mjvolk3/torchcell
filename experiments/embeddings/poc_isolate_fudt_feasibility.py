# experiments/embeddings/poc_isolate_fudt_feasibility.py
# [[experiments.embeddings.poc_isolate_fudt_feasibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/embeddings/poc_isolate_fudt_feasibility
"""WS9b feasibility proof-of-concept: per-isolate FUDT windows from CDS->assembly alignment.

For a few Caudal/Peter natural isolates and a small gene set, this script:

  1. Extracts each isolate's per-gene CDS (Peter gene-keyed store) AND its own de-novo
     assembly (``1011Assemblies.tar.gz``).
  2. BLASTNs every CDS against the isolate's own assembly and keeps the best hit -- measuring
     identity, coverage, and unique-hit margin (alignment reliability).
  3. Slices the 5'/3' FUDT windows from the assembly around each hit -- measuring the
     contig-end / insufficient-flank rate (how often a full window is unavailable).
  4. Round-trips >=1 window through ``FungalUpDownTransformer`` to confirm the pipeline
     produces embeddings, printing the embedding shape.

It writes a JSON summary to ``experiments/embeddings/results/`` and does NOT run the full
943-isolate compute. BLAST+ must be resolvable (``$TC_BLASTN`` / ``$TC_MAKEBLASTDB`` or PATH).

Run from repo root:
  env PYTHONPATH=$(pwd) TC_BLASTN=... TC_MAKEBLASTDB=... \
      ~/miniconda3/envs/torchcell/bin/python \
      experiments/embeddings/poc_isolate_fudt_feasibility.py \
      --isolates AAA AAB AAC --n-genes 40
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import statistics

from dotenv import load_dotenv

from experiments.embeddings.compute_isolate_embeddings import (
    ASSEMBLY_TAR_REL,
    FUDT_MIN_DOWNSTREAM_BP,
    REFGENE_TAR_REL,
    _load_fudt,
    _read_fasta_seqs,
    build_blast_db,
    extract_isolate_assemblies,
    extract_multi_isolate_orfs,
    locate_cds_in_assembly,
    resolve_blast_tools,
    slice_fudt_windows,
)

EXPERIMENT_RESULTS_REL = "experiments/embeddings/results"


def main() -> None:
    """Run the WS9b per-isolate FUDT feasibility PoC and write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolates", nargs="+", default=["AAA", "AAB", "AAC"])
    parser.add_argument(
        "--n-genes", type=int, default=40, help="Gene files scanned (PoC cap)."
    )
    parser.add_argument(
        "--embed", action="store_true", help="Round-trip windows through FUDT (heavy)."
    )
    args = parser.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    cds_tar = osp.join(data_root, REFGENE_TAR_REL)
    asm_tar = osp.join(data_root, ASSEMBLY_TAR_REL)
    blastn, makeblastdb = resolve_blast_tools()
    work = osp.join(data_root, "data/scerevisiae/caudal2024_isolate_embeddings/_poc_ws9b")
    os.makedirs(work, exist_ok=True)

    print(f"extracting CDS for {args.isolates} (cap {args.n_genes} genes)...")
    orfs_by_iso = extract_multi_isolate_orfs(cds_tar, args.isolates, limit=args.n_genes)
    print("extracting assemblies (one forward pass)...")
    asm_by_iso = extract_isolate_assemblies(asm_tar, args.isolates, osp.join(work, "asm"))

    report: dict[str, object] = {
        "isolates": args.isolates,
        "n_genes_cap": args.n_genes,
        "blastn": blastn,
        "per_isolate": {},
    }
    example_windows: list[tuple[str, str, str, str]] = []  # (iso, gene, 5', 3')

    for iso in args.isolates:
        orfs = orfs_by_iso[iso]
        asm_path = asm_by_iso[iso]
        db_path = build_blast_db(asm_path, osp.join(work, "db"), makeblastdb)
        contigs = _read_fasta_seqs(asm_path)
        hits = locate_cds_in_assembly(orfs, db_path, blastn)

        idents: list[float] = []
        coverages: list[float] = []
        unique_margins: list[float] = []
        five_full = three_full = 0
        five_trunc = three_trunc = 0
        unembeddable_3p = 0
        unmapped = 0
        for gene in orfs:
            hit = hits.get(gene)
            if hit is None:
                unmapped += 1
                continue
            idents.append(hit.pident)
            coverages.append(hit.coverage)
            if hit.second_bitscore:
                unique_margins.append(hit.bitscore - hit.second_bitscore)
            sl = slice_fudt_windows(contigs[hit.contig], hit)
            five_full += int(sl.five_prime_full)
            three_full += int(sl.three_prime_full)
            five_trunc += int(not sl.five_prime_full)
            three_trunc += int(not sl.three_prime_full)
            if len(sl.three_prime) < FUDT_MIN_DOWNSTREAM_BP:
                unembeddable_3p += 1
            if len(example_windows) < 3 and sl.five_prime_full and sl.three_prime_full:
                example_windows.append(
                    (iso, gene, sl.five_prime, sl.three_prime)
                )

        n_mapped = len(orfs) - unmapped
        iso_rep = {
            "n_genes": len(orfs),
            "n_mapped": n_mapped,
            "n_unmapped": unmapped,
            "assembly": osp.basename(asm_path),
            "n_contigs": len(contigs),
            "mean_pident": round(statistics.mean(idents), 3) if idents else None,
            "min_pident": round(min(idents), 3) if idents else None,
            "mean_coverage": round(statistics.mean(coverages), 4) if coverages else None,
            "n_full_length_100pct": sum(
                1 for i, c in zip(idents, coverages) if i == 100.0 and c >= 0.999
            ),
            "median_unique_bitscore_margin": (
                round(statistics.median(unique_margins), 1) if unique_margins else None
            ),
            "n_second_hit": len(unique_margins),
            "five_prime_full": five_full,
            "five_prime_truncated": five_trunc,
            "three_prime_full": three_full,
            "three_prime_truncated": three_trunc,
            "three_prime_unembeddable_lt12bp": unembeddable_3p,
            "insufficient_flank_rate": round(
                (five_trunc + three_trunc) / (2 * n_mapped), 4
            )
            if n_mapped
            else None,
        }
        report["per_isolate"][iso] = iso_rep  # type: ignore[index]
        print(f"  {iso}: {iso_rep}")

    # Round-trip >=1 window through FUDT to confirm embeddings.
    if args.embed and example_windows:
        iso, gene, five_seq, three_seq = example_windows[0]
        up = _load_fudt("species_upstream")
        down = _load_fudt("species_downstream")
        up_emb = up.embed([five_seq], mean_embedding=True)
        down_emb = down.embed([three_seq], mean_embedding=True)
        report["embedding_roundtrip"] = {
            "isolate": iso,
            "gene": gene,
            "five_prime_len": len(five_seq),
            "three_prime_len": len(three_seq),
            "upstream_embedding_shape": list(up_emb.shape),
            "downstream_embedding_shape": list(down_emb.shape),
        }
        print(f"  FUDT round-trip {iso}/{gene}: up {tuple(up_emb.shape)} "
              f"down {tuple(down_emb.shape)}")

    os.makedirs(EXPERIMENT_RESULTS_REL, exist_ok=True)
    out = osp.join(EXPERIMENT_RESULTS_REL, "poc_isolate_fudt_feasibility.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
