# experiments/036-dataset-fixes-before-kg-build/scripts/artifact_ref_loader_slice.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.artifact_ref_loader_slice]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/artifact_ref_loader_slice
"""Phase 3 of the artifact tier: the ``ArtifactRef``s the Caudal and Bloom loaders now build.

No dataset is rebuilt. For Caudal, the first two isolates (sorted) of the dev tree's cached
variant table (``preprocess/sequence_variants.parquet``, read-only) are turned into
``SequenceVariantPerturbation``s through the loader's own ``refgene_tarball_ref`` and
``_sequence_perturbations``; for Bloom, the parents of crosses A and 375 are built through
``ParentAssemblyRefs.from_tier`` and ``build_parent``. Every ref is round-tripped through
``ArtifactRef.parse(str(ref), sha256=...)`` and checked with
``torchcell.artifacts.check`` (manifest lookup in the local genomes tier, no download).
Writes ``results/artifact_ref_loader_slice.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

from torchcell.artifacts import check
from torchcell.datamodels.schema import ArtifactRef
from torchcell.datasets.scerevisiae import bloom2019 as b
from torchcell.datasets.scerevisiae import caudal2024 as c

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "artifact_ref_loader_slice.json",
)


def _describe(ref: ArtifactRef, data_root: str) -> dict[str, object]:
    """The ref's string form, pin, round-trip result and local-tier check."""
    return {
        "tc": str(ref),
        "sha256": ref.sha256,
        "bytes": ref.bytes,
        "parse_round_trip": ArtifactRef.parse(
            str(ref), sha256=ref.sha256, bytes=ref.bytes
        )
        == ref,
        "resolves_locally": check(ref, data_root=data_root),
    }


def main() -> None:
    """Build the slice and write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    variants = pd.read_parquet(
        osp.join(
            data_root,
            "data/torchcell/caudal_pantranscriptome2024/preprocess/sequence_variants.parquet",
        )
    )
    strains = sorted(variants["strain_id"].unique())[:2]
    tarball = c.refgene_tarball_ref(data_root)
    caudal: dict[str, object] = {}
    for strain in strains:
        sub = variants[variants["strain_id"] == strain]
        rows = list(
            zip(sub["systematic_gene_name"], sub["symbol"], sub["header_token"])
        )
        perts = c.CaudalPanTranscriptome2024Dataset._sequence_perturbations(
            strain, rows, tarball
        )
        refs = [p.sequence_ref for p in perts if p.sequence_ref is not None]
        caudal[strain] = {
            "n_sequence_variants": len(perts),
            "n_with_ref": len(refs),
            "first_two": [_describe(r, data_root) for r in refs[:2]],
            "all_round_trip": all(
                ArtifactRef.parse(str(r), sha256=r.sha256, bytes=r.bytes) == r
                for r in refs
            ),
        }
    assemblies = b.ParentAssemblyRefs.from_tier(data_root)
    info = b.read_cross_table(
        osp.join(data_root, "data/torchcell/bloom2019/raw", b.XLS_NAME),
        osp.join(data_root, "data/torchcell/bloom2019/raw", b.README_NAME),
    )
    bloom: dict[str, object] = {}
    for cross in ("A", "375"):
        cross_info = info[cross]
        for label, text in (
            (cross_info.parent_1, cross_info.parent_1_genotype),
            (cross_info.parent_2, cross_info.parent_2_genotype),
        ):
            parent = b.build_parent(label, text, assemblies)
            bloom[f"{cross}/{label}"] = _describe(parent.assembly_ref, data_root)
    out = {"caudal": caudal, "bloom": bloom}
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
