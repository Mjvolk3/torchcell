# experiments/043-expression-proteome-morphology/scripts/embed_vaishnav_reporter_043.py
# [[experiments.043-expression-proteome-morphology.scripts.embed_vaishnav_reporter_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/embed_vaishnav_reporter_043.py
"""The Vaishnav 2022 promoter windows and the yEVenus reporter into the allele tables.

The reporter token of the primer (``input-design.tex``, Eq. input-reporter) reads three
of the four composite encoders: FUDT upstream on the 1,003 bp window ending at the
reporter's start codon (one per released variant, 71,158 variants), ProtT5 on the
reporter protein and CaLM on the reporter CDS. The FUDT downstream window is one
constant 300 bp backbone sequence and is embedded too (one row), so the reporter carries
all four keys like any gene.

Every row goes into the SAME tables the isolate alleles use,
``$DATA_ROOT/data/scerevisiae/isolate_embeddings/peter2018_1011/alleles/<encoder>.h5``,
keyed by the sha256 of the exact string the encoder reads, with source ``vaishnav2022``
or ``reporter``. The batch path, the reference check and the fp16 storage are those of
``compute_isolate_composite.py`` (imported, not copied), so a promoter window and a native
promoter go through one function. That script is not edited: job 6106 reads it at run
time.

Stages:

``inputs``  CPU. Rebuild every variant's windows from the Addgene 127546 contig with
            ``vaishnav_promoter_window.py``'s functions, check each sha256 against
            ``results/vaishnav_promoter_windows.parquet``, extract the reporter CDS and
            protein, and write ``sequences/vaishnav2022_inputs.parquet`` (sha256, kind,
            encoder, sequence, source) beside the isolate sequence tables.
``embed``   GPU. Per encoder, the rows of that table not yet in its h5 table; the reference
            check of ``compute_isolate_composite.py`` runs first and stops on disagreement.
``verify``  CPU. Every input key has a row in its table; writes
            ``results/embed_vaishnav_reporter_043.json``.

    python experiments/043-expression-proteome-morphology/scripts/embed_vaishnav_reporter_043.py --stage inputs
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import compute_isolate_composite as cic  # noqa: E402
import vaishnav_promoter_window as vpw  # noqa: E402

INPUTS_REL = "sequences/vaishnav2022_inputs.parquet"
RESULTS_JSON = "embed_vaishnav_reporter_043.json"
SourceName = Literal["vaishnav2022", "reporter"]


class InputRow(BaseModel):
    """One sequence one encoder reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    kind: cic.InputKind
    encoder: cic.EncoderName
    sequence: str
    source: SourceName
    label: str = Field(description="variant id, or the reporter feature name")


class ReporterInputs(BaseModel):
    """The reporter's sequences, read out of the deposited contig."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_label: str
    cds: str
    protein: str
    calm: str
    downstream_window: str


def results_dir() -> Path:
    """``experiments/043-.../results``."""
    return Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"


def reporter_inputs(
    site: vpw.SiteGeometry, backbone: str, config: vpw.RunConfig, down: str
) -> ReporterInputs:
    """The reporter CDS (deposited coordinates; the insert does not touch it)."""
    cds = backbone[site.reporter_start - 1 : site.reporter_end].upper()
    if cds[:3] != "ATG" or len(cds) % 3:
        raise RuntimeError(
            f"reporter CDS is not a whole ORF ({len(cds)} bp, {cds[:3]})"
        )
    protein = cic.translate(cds)
    if not protein.endswith("*") or "*" in protein[:-1]:
        raise RuntimeError("reporter protein has an internal stop or no stop")
    return ReporterInputs(
        feature_label=config.reporter_feature_label,
        cds=cds,
        protein=protein,
        calm=cic.calm_input(cds, cic.WindowGeometry().calm_max_nt),
        downstream_window=down,
    )


def stage_inputs(data_root: str, out_root: Path) -> None:
    """Rebuild and check every window; write the input table."""
    t0 = time.time()
    config = vpw.RunConfig()
    site, backbone = vpw.geometry(config, data_root)
    records = vpw.read_inserts(config, data_root)
    released = pd.read_parquet(results_dir() / "vaishnav_promoter_windows.parquet")
    if list(released.variant_id) != list(records.variant_id):
        raise RuntimeError(
            "variant order differs from vaishnav_promoter_windows.parquet"
        )
    rows: dict[tuple[str, str], InputRow] = {}
    downs: set[str] = set()
    for vid, insert, up_sha, down_sha in zip(
        records.variant_id,
        records.insert_80mer,
        released.upstream_window_sha256,
        released.downstream_window_sha256,
        strict=True,
    ):
        up, down = vpw.windows(insert, backbone, site, config.window)
        up, down = up.upper(), down.upper()
        if cic.sha256_text(up) != up_sha or cic.sha256_text(down) != down_sha:
            raise RuntimeError(f"{vid}: rebuilt window differs from the released table")
        rows.setdefault(
            (up_sha, "fudt_species_upstream"),
            InputRow(
                sha256=up_sha,
                kind="window_up",
                encoder="fudt_species_upstream",
                sequence=up,
                source="vaishnav2022",
                label=vid,
            ),
        )
        downs.add(down)
    if len(downs) != 1:
        raise RuntimeError(f"{len(downs)} distinct downstream windows, expected 1")
    rep = reporter_inputs(site, backbone, config, downs.pop())
    for kind, encoder, seq in (
        ("window_down", "fudt_species_downstream", rep.downstream_window),
        ("protein", "prot_t5_xl_uniref50", rep.protein),
        ("calm", "calm", rep.calm),
    ):
        sha = cic.sha256_text(seq)
        rows[(sha, encoder)] = InputRow(
            sha256=sha,
            kind=kind,
            encoder=encoder,
            sequence=seq,
            source="reporter",
            label=rep.feature_label,
        )
    frame = pd.DataFrame([r.model_dump() for r in rows.values()])
    cic.write_parquet(frame, out_root / INPUTS_REL)
    summary = {
        "variants": len(records),
        "rows": len(frame),
        "rows_per_encoder": frame.groupby("encoder").size().to_dict(),
        "reporter": {
            "feature_label": rep.feature_label,
            "cds_bp": len(rep.cds),
            "protein_aa_with_stop": len(rep.protein),
            "cds_sha256": cic.sha256_text(rep.cds),
            "protein_sha256": cic.sha256_text(rep.protein),
            "calm_sha256": cic.sha256_text(rep.calm),
            "downstream_window_sha256": cic.sha256_text(rep.downstream_window),
        },
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=1))
    (out_root / "sequences" / "vaishnav2022_inputs.json").write_text(
        json.dumps(summary, indent=1)
    )


def stage_embed(cfg: cic.IsolateCompositeConfig, data_root: str) -> None:
    """Per encoder, the input rows not yet in its table (fp16)."""
    import torch

    frame = pd.read_parquet(cfg.out_root / INPUTS_REL)
    for name in (
        "fudt_species_upstream",
        "fudt_species_downstream",
        "calm",
        "prot_t5_xl_uniref50",
    ):
        t0, started = time.time(), cic.now()
        spec = cic.ENCODERS[name]
        path = cic.h5_path(cfg, name)
        have = cic.h5_keys(path)
        sub = frame[frame["encoder"] == name]
        todo = [
            (k, s)
            for k, s in zip(sub["sha256"], sub["sequence"], strict=True)
            if k not in have
        ]
        print(f"embed {name}: {len(sub)} inputs, {len(todo)} to compute", flush=True)
        if not todo:
            continue
        emb = cic.Embedder(spec, cfg.device)
        check = cic.reference_check(cfg, data_root, emb)
        plan = cic.batches(cfg, emb, todo)
        buf_k: list[str] = []
        buf_v: list[np.ndarray] = []
        n_done = 0
        source = str(sub["source"].iloc[0])
        for bi, b in enumerate(plan, start=1):
            vecs = emb([s for _, s in b])
            buf_k += [k for k, _ in b]
            buf_v.append(vecs)
            if bi % cfg.flush_every_batches == 0 or bi == len(plan):
                cic.h5_append(path, buf_k, np.concatenate(buf_v), spec.dim, source)
                n_done += len(buf_k)
                buf_k, buf_v = [], []
                el = time.time() - t0
                print(
                    f"embed {name}: {n_done}/{len(todo)} in {el:.0f} s "
                    f"({n_done / max(el, 1e-9):.1f} per s)",
                    flush=True,
                )
        del emb
        torch.cuda.empty_cache()
        cic.record_stage(
            cfg,
            f"embed_vaishnav_reporter_{name}",
            t0,
            started,
            counts={
                "inputs": len(sub),
                "computed": len(todo),
                "batches": len(plan),
                **{f"reference_{k}": v for k, v in check.items()},
            },
        )


def stage_verify(cfg: cic.IsolateCompositeConfig) -> None:
    """Every input key has a row in its encoder's table."""
    frame = pd.read_parquet(cfg.out_root / INPUTS_REL)
    report: dict[str, object] = {
        "generated_by": "experiments/043-expression-proteome-morphology/scripts/"
        "embed_vaishnav_reporter_043.py",
        "created_at": cic.now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "encoders": {},
    }
    complete = True
    for name, sub in frame.groupby("encoder"):
        keys = cic.h5_keys(cic.h5_path(cfg, str(name)))
        n_have = int(sub["sha256"].isin(keys).sum())
        report["encoders"][str(name)] = {  # type: ignore[index]
            "inputs": len(sub),
            "with_row": n_have,
            "table_rows": len(keys),
        }
        complete &= n_have == len(sub)
    report["complete"] = complete
    cic.write_json(report, results_dir() / RESULTS_JSON)
    print(json.dumps(report, indent=1))
    if not complete:
        raise RuntimeError("some Vaishnav or reporter inputs have no embedding row")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=["inputs", "embed", "verify"])
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    cfg = cic.build_config(data_root, None)
    if args.stage == "inputs":
        stage_inputs(data_root, cfg.out_root)
    elif args.stage == "embed":
        stage_embed(cfg, data_root)
    else:
        stage_verify(cfg)


if __name__ == "__main__":
    main()
