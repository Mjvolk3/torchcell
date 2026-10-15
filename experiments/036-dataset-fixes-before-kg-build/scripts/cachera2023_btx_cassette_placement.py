# experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_btx_cassette_placement.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_btx_cassette_placement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_btx_cassette_placement
"""Measure every number the Cachera 2023 Btx-cassette placement pins (P4).

The four ``GeneAdditionPerturbation`` records of ``betaxanthin_cachera2023`` carry a
``ConstructPlacement`` into the deposited pBTX1 map. Every value in that placement is
measured here rather than read off a design note:

1. WHICH contig. The deposited construct set ``cachera2023_cri_spa_plasmids`` holds 6
   indexed contigs and 11 blocked ones; pBTX1 is indexed. The paper states the screened
   strains' cassette came from pBTX1 ("CD-Btx was constructed ... by transforming with
   the integrative plasmid pBTX1"), and the CRI-SPA screen transfers that cassette from
   CD-Btx into the YKO library, so pBTX1 is the contig the records point at. pBTX2 is
   the BY-Btx vector and is one of the 11 maps blocked by an origin-wrapping location.
2. WHICH feature. The four gene features of pBTX1 by label, SO id and 1-based span.
3. WHICH coordinate a designed allele sits at. Each gene feature is checked to be a
   CDS in frame (length divisible by three, ATG start, one terminal stop, no internal
   stop) by translating the bytes, and the contig coordinates of ARO4 codon 229 and
   ARO7 codon 141 are derived from the feature span and the codon number. The codon the
   map carries is printed: a derivation is only honest if the construct's own bases show
   the designed residue.
4. WHY ``insert_span`` is a typed gap. The two homology-arm features are printed with
   the three candidate intervals they admit and the paper's own approximate cassette
   size, which do not agree, so the loader states no interval.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/cachera2023_btx_cassette_placement.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_btx_cassette_placement.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq
from dotenv import load_dotenv

from torchcell.sequence.construct.registry import (
    CACHERA2023_CRI_SPA_PLASMIDS,
    load_construct_manifest,
)

#: The contig the records point at, by the name the paper gives it.
SOURCE_NAME = "pBTX1"

#: The four Btx-cassette genes, each as (the loader's gene key, the feature label).
GENES: tuple[tuple[str, str], ...] = (
    ("CYP76AD1", "CYP76AD/P450 fusion"),
    ("DOD", "DOPA 4,5 dioxygenase"),
    ("YBR249C", "ARO4-K229L"),
    ("YPR060C", "ARO7-G141S"),
)

#: The designed alleles the paper states, as (feature label, codon number, residue).
ALLELES: tuple[tuple[str, int, str], ...] = (
    ("ARO4-K229L", 229, "L"),
    ("ARO7-G141S", 141, "S"),
)

#: The homology-arm features that flank the transferred segment.
ARMS: tuple[str, ...] = ("XII-5 up B", "XII-5 dw A")

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "cachera2023_btx_cassette_placement.json",
)


def _library_path(data_root: str, key: str, rel: str) -> str:
    """The mirrored artifact's absolute path in the library tier."""
    return osp.join(data_root, "torchcell-library", key, rel)


def main() -> None:
    """Measure and record the placement's values."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    manifest = load_construct_manifest(CACHERA2023_CRI_SPA_PLASMIDS)
    contigs = [
        contig
        for contig in manifest.contigs
        if contig.source_name is not None and contig.source_name.value == SOURCE_NAME
    ]
    if len(contigs) != 1:
        raise RuntimeError(
            f"{SOURCE_NAME} names {len(contigs)} indexed contigs of "
            f"{CACHERA2023_CRI_SPA_PLASMIDS}; the placement needs exactly one"
        )
    contig = contigs[0]
    path = _library_path(data_root, contig.ref.key, contig.ref.path)
    record = SeqIO.read(path, "genbank")
    sequence = str(record.seq)
    if len(sequence) != contig.length:
        raise RuntimeError(
            f"{path} is {len(sequence)} bp but the index records {contig.length}"
        )

    by_label: dict[str, Any] = {}
    for feature in contig.features:
        by_label.setdefault(feature.key.label, []).append(feature)

    out: dict[str, Any] = {
        "construct_set": manifest.construct_set,
        "sequence_status": str(manifest.sequence_status),
        "n_indexed": len(manifest.contigs),
        "n_blocked": len(manifest.blocked),
        "blocked_paths": [blocked.path for blocked in manifest.blocked],
        "contig": {
            "handle": contig.handle,
            "source_name": SOURCE_NAME,
            "source_quote": contig.source_name.quote
            if contig.source_name is not None
            else None,
            "record_name": contig.record_name,
            "length": contig.length,
            "topology": contig.topology,
            "ref": contig.ref.model_dump(),
        },
        "genes": [],
        "alleles": [],
        "arms": [],
    }

    frames: dict[str, str] = {}
    for gene_key, label in GENES:
        features = [
            feature
            for feature in by_label.get(label, [])
            if feature.genbank_type == "gene"
        ]
        if len(features) != 1:
            raise RuntimeError(f"{label}: {len(features)} gene features, expected 1")
        key = features[0].key
        coding = sequence[key.span.start - 1 : key.span.end]
        if key.span.orientation == "reverse_complement":
            coding = str(Seq(coding).reverse_complement())
        frames[label] = coding
        protein = str(Seq(coding).translate())
        out["genes"].append(
            {
                "gene_key": gene_key,
                "token": key.token,
                "label": key.label,
                "so_id": key.so_id,
                "start": key.span.start,
                "end": key.span.end,
                "orientation": key.span.orientation,
                "length": len(coding),
                "in_frame": len(coding) % 3 == 0,
                "starts_atg": coding[:3] == "ATG",
                "terminal_stop": protein.endswith("*"),
                "internal_stop": "*" in protein[:-1],
                "n_residues": len(protein.rstrip("*")),
            }
        )

    for label, codon_number, residue in ALLELES:
        coding = frames[label]
        offset = (codon_number - 1) * 3
        codon = coding[offset : offset + 3]
        key = next(
            feature.key for feature in by_label[label] if feature.genbank_type == "gene"
        )
        if key.span.orientation == "reverse_complement":
            start, end = key.span.end - offset - 2, key.span.end - offset
        else:
            start, end = key.span.start + offset, key.span.start + offset + 2
        out["alleles"].append(
            {
                "label": label,
                "codon_number": codon_number,
                "codon_on_contig": codon,
                "translates_to": str(Seq(codon).translate()),
                "expected_residue": residue,
                "carries_designed_residue": str(Seq(codon).translate()) == residue,
                "contig_position_start": start,
                "contig_position_end": end,
            }
        )

    arms = []
    for label in ARMS:
        key = next(feature.key for feature in by_label[label])
        arms.append({"label": label, "start": key.span.start, "end": key.span.end})
    out["arms"] = arms
    first, second = arms[0], arms[1]
    out["insert_span_candidates"] = {
        "arm_to_arm_inclusive_bp": second["end"] - first["start"] + 1,
        "between_arms_exclusive_bp": second["start"] - first["end"] - 1,
        "paper_statement": "the considerable size, 9.3 kb, of the Btx-cassette",
        "agree": False,
    }

    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
