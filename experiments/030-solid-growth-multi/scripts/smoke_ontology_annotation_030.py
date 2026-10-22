# experiments/030-solid-growth-multi/scripts/smoke_ontology_annotation_030.py
# [[experiments.030-solid-growth-multi.scripts.smoke_ontology_annotation_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/smoke_ontology_annotation_030
"""The data-level pass condition of the ontology conditioning, on a real store.

plan.030-ontology-conditioning.2026.10.22: before any ontology arm is read, the
annotation itself must show that (1) the two Kuzmin trigenic screens of one measurement
type annotate IDENTICALLY, so no token can tell them apart, (2) Costanzo and Kuzmin
interaction rows differ only by the uracil dropout and the temperature, the two record
facts that separate the plates, and (3) fitness and interaction rows of one screen
differ in their phenotype and experiment classes, (4) the two converted sources (SGD
essentiality, SynthLethDb) differ by their origin class (``CONVERTED_FROM``) and nothing
else. Reads ``per_dataset`` records of every
source straight from the store, builds the vocabulary exactly as the training script
does, prints every source's distinct profiles and exits non-zero on a failed check.

    python experiments/030-solid-growth-multi/scripts/smoke_ontology_annotation_030.py \
        --store $DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3q
"""

from __future__ import annotations

import argparse
import sys

import torch

from torchcell.data.experiment_ontology import (
    OntologyDagEncoder,
    build_experiment_ontology_vocabulary,
    describe_annotation,
    scan_store_records,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--per-dataset", type=int, default=25)
    ap.add_argument("--write-vocab", default=None, help="write the vocabulary JSON here")
    args = ap.parse_args()

    scanned = scan_store_records(args.store, per_dataset=args.per_dataset)
    vocab = build_experiment_ontology_vocabulary((e, r) for _, e, r in scanned)
    depths = vocab.depths()
    print(
        f"vocabulary: {len(vocab)} nodes, max depth {max(depths)}, root depth "
        f"{depths[vocab.index(vocab.root)]}; {len(scanned)} records over "
        f"{len({n for n, _, _ in scanned})} sources"
    )
    profiles: dict[str, set[tuple[str, ...]]] = {}
    for name, e, r in scanned:
        profiles.setdefault(name, set()).add(
            tuple(describe_annotation(vocab, vocab.vector(e, r)))
        )
    for name in sorted(profiles):
        print(f"  {name}: {len(profiles[name])} distinct profile(s)")

    failures: list[str] = []

    def one(name: str) -> set[str]:
        return set(next(iter(profiles[name])))

    # (1) the two Kuzmin screens of one measurement type are indistinguishable
    for a, b in (
        ("TmiKuzmin2018Dataset", "TmiKuzmin2020Dataset"),
        ("TmfKuzmin2018Dataset", "TmfKuzmin2020Dataset"),
        ("DmiKuzmin2018Dataset", "DmiKuzmin2020Dataset"),
    ):
        if a in profiles and b in profiles:
            diff = one(a) ^ one(b)
            print(f"{a} vs {b}: {'identical' if not diff else sorted(diff)}")
            if diff:
                failures.append(f"{a} and {b} differ: {sorted(diff)}")
    # (2) Costanzo vs Kuzmin interaction rows differ by the uracil dropout and the
    # temperature value only
    if "DmiCostanzo2016Dataset" in profiles and "DmiKuzmin2018Dataset" in profiles:
        allowed_prefix = ("Media.dropout:uracil", "Temperature.value=")
        for cost in profiles["DmiCostanzo2016Dataset"]:
            diff = set(cost) ^ one("DmiKuzmin2018Dataset")
            bad = [d for d in diff if not d.startswith(allowed_prefix)]
            print(f"DmiCostanzo2016 vs DmiKuzmin2018: {sorted(diff)}")
            if bad:
                failures.append(f"Costanzo vs Kuzmin differ beyond uracil/temperature: {bad}")
    # (3) fitness vs interaction rows of one screen differ in their classes
    if "TmfKuzmin2018Dataset" in profiles and "TmiKuzmin2018Dataset" in profiles:
        diff = one("TmfKuzmin2018Dataset") ^ one("TmiKuzmin2018Dataset")
        print(f"TmfKuzmin2018 vs TmiKuzmin2018: {sorted(diff)}")
        if not any("Phenotype" in d for d in diff):
            failures.append("fitness and interaction rows share their phenotype class")
    # (4) the two converted sources differ by their origin class and nothing else
    if (
        "GeneEssentialitySgdDataset" in profiles
        and "SynthLethalityYeastSynthLethDbDataset" in profiles
    ):
        diff = one("GeneEssentialitySgdDataset") ^ one(
            "SynthLethalityYeastSynthLethDbDataset"
        )
        print(f"GeneEssentialitySgd vs SynthLethDb: {sorted(diff)}")
        if diff != {
            "Experiment.converted_from:GeneEssentialityExperiment",
            "Experiment.converted_from:SyntheticLethalityExperiment",
        }:
            failures.append(f"converted sources differ beyond their origin: {sorted(diff)}")
    if args.write_vocab:
        vocab.to_json(args.write_vocab)
        print(f"vocabulary written: {args.write_vocab}")
    # the encoder runs and separates what the annotation separates
    enc = OntologyDagEncoder(vocab, state_dim=8, token_dim=8)
    x = torch.stack([vocab.vector(e, r) for _, e, r in scanned])
    tok = enc(x)
    names = [n for n, _, _ in scanned]
    tmi18 = [i for i, n in enumerate(names) if n == "TmiKuzmin2018Dataset"]
    tmi20 = [i for i, n in enumerate(names) if n == "TmiKuzmin2020Dataset"]
    if tmi18 and tmi20 and not torch.allclose(tok[tmi18[0]], tok[tmi20[0]], atol=1e-6):
        failures.append("encoder tokens differ for identical annotations")
    print(f"encoder: token {tuple(tok.shape)}, {sum(p.numel() for p in enc.parameters())} parameters")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
