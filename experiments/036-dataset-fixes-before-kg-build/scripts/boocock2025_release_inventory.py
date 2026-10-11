# experiments/036-dataset-fixes-before-kg-build/scripts/boocock2025_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.boocock2025_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/boocock2025_release_inventory.py
"""Measure the Boocock 2025 release before and for the loader: what each panel holds,
what the paper's printed numbers reproduce to, and the genotype cardinality cost.

Reads the four ``segData.RDS``, ``segMatch.RDS`` and ``gdata_42k.RData`` from the raw
mirror (each sha256-checked against the mirror's ``manifest.json``) and writes
``experiments/036-dataset-fixes-before-kg-build/results/boocock2025_release_inventory.json``:

- per panel: cells, analysis transcripts, median UMI, libraries, cell-cycle stages,
  LD-pruned markers, the fraction of marker posteriors strictly inside (0.05, 0.95), and
  the mean P(parent 2) at the marker nearest MAT;
- the cardinality table the eQTL data-model note asks for: blocks per record and pickled
  genotype bytes per record for a pure called-parent run and for posterior grids of 0.1,
  0.05, 0.02 and 0.01, on 500 cells per de novo panel drawn with seed 0;
- the pooled panel: segregants, cells per segregant, the per-cell agreement with the
  whole-genome calls on the shared markers, and the lookup genotypes' block counts.

Run from the repo root: ``python experiments/036-dataset-fixes-before-kg-build/scripts/
boocock2025_release_inventory.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.datasets.scerevisiae import boocock2025 as b
from torchcell.datasets.scerevisiae.bloom2019 import encode_blocks

STEPS: list[float | None] = [None, 0.1, 0.05, 0.02, 0.01]
SAMPLE = 500
MAT_POSITION = 200_000
OUT = "experiments/036-dataset-fixes-before-kg-build/results/boocock2025_release_inventory.json"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blocks_at(
    posterior: np.ndarray[Any, Any], markers: list[Any], step: float | None
) -> list[HaplotypeBlock]:
    """Runs of called parent (step None) or of called parent and grid level."""
    parent = np.where(posterior > 0.5, 2, 1)
    q = np.where(parent == 2, posterior, 1 - posterior)
    level = np.zeros(len(q), dtype=np.int64) if step is None else np.rint(q / step)
    chrom = np.array([m.chromosome for m in markers])
    change = np.ones(len(q), dtype=bool)
    change[1:] = (
        (parent[1:] != parent[:-1])
        | (level[1:] != level[:-1])
        | (chrom[1:] != chrom[:-1])
    )
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(q)) - 1
    out = []
    for s, e in zip(starts, ends):
        if step is None:
            value = float(q[s : e + 1].min())
        else:
            value = round(float(level[s]) * step, 2)
        out.append(
            HaplotypeBlock(
                chromosome=markers[s].chromosome,
                start=markers[s].position,
                end=markers[e].position,
                parent=int(parent[s]),  # type: ignore[arg-type]
                posterior=min(value, 1.0),
                n_markers=int(e - s + 1),
            )
        )
    return out


def _summary(values: list[int] | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 1),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def main() -> None:
    """Measure every panel and write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    mirror = str(b.raw_mirror_dir(data_root))
    manifest = b.load_manifest(data_root)
    rel = b.raw_relpaths()
    paths = {}
    for name in rel:
        path = osp.join(mirror, rel[name])
        if _sha256(path) != b.manifest_sha256(manifest, rel[name]):
            raise RuntimeError(f"{path} differs from its manifest pin")
        paths[name] = path
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"panels": {}, "sample_cells_per_panel": SAMPLE}
    for spec in b.PANELS.values():
        release = b.read_panel_release(paths[spec.segdata_name])
        on_iii = [i for i, m in enumerate(release.markers) if m.chromosome == "chrIII"]
        mat = min(on_iii, key=lambda i: abs(release.markers[i].position - MAT_POSITION))
        x = release.posterior
        panel: dict[str, Any] = {
            "cells": len(release.barcodes),
            "transcripts": len(release.transcripts),
            "median_umi": float(np.median(release.n_umi)),
            "libraries": dict(Counter(release.libraries)),
            "stages": dict(Counter(release.stages)),
            "markers": len(release.markers),
            "fraction_posterior_uncertain": round(
                float(np.mean((x > 0.05) & (x < 0.95))), 4
            ),
            "mat_marker": f"{release.markers[mat].chromosome}:{release.markers[mat].position}",
            "mat_mean_p_parent_2": round(float(np.mean(x[:, mat])), 4),
        }
        if not spec.pooled:
            rows = rng.choice(x.shape[0], min(SAMPLE, x.shape[0]), replace=False)
            cardinality = {}
            for step in STEPS:
                blocks = [blocks_at(x[r], release.markers, step) for r in rows]
                cardinality["called_parent_only" if step is None else str(step)] = {
                    "blocks": _summary([len(bl) for bl in blocks]),
                    "genotype_bytes": _summary(
                        [
                            len(pickle.dumps([bk.model_dump() for bk in bl]))
                            for bl in blocks
                        ]
                    ),
                }
            panel["cardinality"] = cardinality
        else:
            match = b.read_segregant_match(paths[b.SEGMATCH_NAME])
            ids, markers, calls = b.read_lookup_genotypes(paths[b.GDATA_NAME])
            per = Counter(match.segregant)
            key = {
                (m.chromosome, m.position, m.ref, m.alt): j
                for j, m in enumerate(markers)
            }
            pairs = [
                (i, key[(m.chromosome, m.position, m.ref, m.alt)])
                for i, m in enumerate(release.markers)
                if (m.chromosome, m.position, m.ref, m.alt) in key
            ]
            row = {s: i for i, s in enumerate(ids)}
            hmm = x[:, [a for a, _ in pairs]] > 0.5
            wgs = calls[[row[s] for s in match.segregant]][:, [c for _, c in pairs]] > 0
            agreement = (hmm == wgs).mean(axis=1)
            pooled_blocks = [
                len(
                    encode_blocks(
                        np.where(calls[row[s]] > 0, 2, 1).astype(np.int8), markers
                    )
                )
                for s in sorted(per)
            ]
            panel["pooled"] = {
                "lookup_segregants": len(ids),
                "lookup_markers": len(markers),
                "matched_segregants": len(per),
                "median_cells_per_segregant": float(np.median(list(per.values()))),
                "segregants_over_ten_cells": int(sum(v > 10 for v in per.values())),
                "match_correlation": _summary(match.correlation),
                "shared_markers": len(pairs),
                "median_agreement": round(float(np.median(agreement)), 4),
                "wgs_blocks_per_segregant": _summary(pooled_blocks),
            }
        out["panels"][spec.panel] = panel
        print(
            spec.panel,
            json.dumps({k: v for k, v in panel.items() if k != "cardinality"}),
        )
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {OUT}")


def render(report_path: str) -> str:
    """Markdown tables for the dendron note, from the inventory JSON and a verification
    report (``preprocess/verification_report.json`` of the built store).
    """
    with open(OUT) as handle:
        inventory = json.load(handle)
    lines = [
        "| panel | cells | transcripts | median UMI | LD-pruned markers | "
        "posteriors in (0.05, 0.95) | P(parent 2) at MAT |",
        "|---|---|---|---|---|---|---|",
    ]
    for panel, v in inventory["panels"].items():
        lines.append(
            f"| {panel} | {v['cells']:,} | {v['transcripts']:,} | {v['median_umi']} | "
            f"{v['markers']:,} | {v['fraction_posterior_uncertain']} | "
            f"{v['mat_mean_p_parent_2']} ({v['mat_marker']}) |"
        )
    lines += [
        "",
        "| panel | encoding | blocks median | blocks p90 | blocks max | "
        "genotype bytes median |",
        "|---|---|---|---|---|---|",
    ]
    for panel, v in inventory["panels"].items():
        for encoding, c in v.get("cardinality", {}).items():
            lines.append(
                f"| {panel} | {encoding} | {c['blocks']['median']:g} | "
                f"{c['blocks']['p90']:g} | {c['blocks']['max']:g} | "
                f"{c['genotype_bytes']['median']:,.0f} |"
            )
    with open(report_path) as handle:
        report = json.load(handle)
    lines += ["", "| level | rule | result | message |", "|---|---|---|---|"]
    for r in sorted(report["results"], key=lambda r: r["level"]):
        message = str(r["message"]).replace("|", "/")
        lines.append(
            f"| L{r['level']} | `{r['name']}` | {'pass' if r['passed'] else 'FAIL'} "
            f"| {message} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "render":
        print(render(sys.argv[2]))
    else:
        main()
