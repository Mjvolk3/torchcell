# experiments/027-betaxanthin-metabolic/scripts/audit_concentration_compartments.py
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_concentration_compartments.py

r"""Which COMPARTMENT of the learned concentration matches the measurement?

The wave-7 concentration readout predicts the whole-cell average
``[A]_total = sum_n f_n [A]_n`` over the compartments yeast-GEM carries each amino acid in,
because that is what Mulleder measures. The mixture hides which compartment the agreement
comes from: a whole-cell number that tracks the measurement because its vacuolar term does,
and one that tracks it because its cytosolic term does, are different claims about the
model, and the fractional volumes that weight them are five parts assumed.

This script re-opens a finished cell from its checkpoint and correlates EACH compartment's
learned concentration, and the mixture, against the measured whole-cell value on the pinned
Merzbacher test genes. Nothing is retrained and no weight is touched; the model is rebuilt
by the trainer's own ``build_model_for_cell`` so the architecture cannot drift from the one
that was scored.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/audit_concentration_compartments.py \
        --result wave7_aa_conc_e200_s2700.json

Writes ``results/conc_compartments_<stem>.json``.
"""

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_bx import (  # noqa: E402
    AA_KEYS,
    AUX_HEAD,
    CHECKPOINT_DIR,
    LN10,
    RESULTS_DIR,
    BxCell,
    _masked_pearson,
    _masked_spearman,
    build_dataset,
    build_model_for_cell,
    code_provenance,
    compartment_volume_fractions,
    load_config,
    mixture_log10_molar,
    mixture_weights,
    pinned_test_mask,
    resolve_pinned_test,
    targets_in_space,
)

from torchcell.datamodules import CellDataModule  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]


class CompartmentAgreement(BaseModel):
    """One (amino acid, compartment) column against the measured whole-cell value."""

    model_config = ConfigDict(extra="forbid")

    amino_acid: str
    compartment: str
    species: str
    fractional_volume: float
    n: int
    spearman: float
    pearson: float
    median_log10_m: float
    #: What this compartment contributes to the whole-cell average, as a fraction of it,
    #: at the median genotype: `f_n c_n / sum_k f_k c_k`.
    median_share_of_mixture: float


class MixtureAgreement(BaseModel):
    """The whole-cell average of one amino acid against its measurement."""

    model_config = ConfigDict(extra="forbid")

    amino_acid: str
    n: int
    spearman: float
    pearson: float
    median_pred_log10_m: float
    median_true_log10_m: float
    best_compartment_by_spearman: str
    best_compartment_spearman: float


class CompartmentAudit(BaseModel):
    """The whole read, written verbatim into the note."""

    model_config = ConfigDict(extra="forbid")

    result_file: str
    checkpoint: str
    arm: str
    seed: int
    epochs: int
    select_epoch: int
    conc_space: str
    conc_lower_m: float
    conc_upper_m: float
    group: str
    volume_fractions: dict[str, Any]
    mixture: list[MixtureAgreement] = Field(default_factory=list)
    compartments: list[CompartmentAgreement] = Field(default_factory=list)
    code_provenance: dict[str, str] = Field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", required=True, help="Result file name under results/."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(osp.join(RESULTS_DIR, args.result)) as fh:
        result = json.load(fh)
    cell = BxCell(**result["cell"])
    assert cell.head == AUX_HEAD, (
        f"this audit is for the amino-acid head; {args.result} trained {cell.head!r}."
    )
    cfg = load_config()
    ckpt_path = args.checkpoint or result.get("checkpoint_path")
    if ckpt_path is None:
        ckpt_path = osp.join(CHECKPOINT_DIR, f"{osp.splitext(args.result)[0]}.pt")
    assert osp.isfile(ckpt_path), f"no checkpoint at {ckpt_path}."
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, "checkpoint is for another seed."

    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head_specs, main_head, flux_meta = build_model_for_cell(
        cell, cfg, dataset, device
    )
    layer: Any = model.flux_layer
    assert layer is not None, "this cell has no flux layer."
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    indices, log_w, described = mixture_weights(flux_meta["met_index"], cell.conc_space)
    indices, log_w = indices.to(device), log_w.to(device)

    model.eval()
    mix_parts: list[np.ndarray] = []
    per_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            log_c = reps["flux"]["thermo_log_c"]
            mix_parts.append(mixture_log10_molar(log_c, indices, log_w).cpu().numpy())
            # [B, 19, C] log10 molar per compartment, padded columns included and dropped
            # later by their -inf weight.
            per = log_c[:, indices.reshape(-1)].reshape(
                log_c.shape[0], indices.shape[0], indices.shape[1]
            )
            per_parts.append((per / LN10).cpu().numpy())
            t, m = targets_in_space(batch, head_specs, True)[AUX_HEAD]
            true_parts.append(torch.where(m, t, torch.nan).cpu().numpy())
    mix = np.concatenate(mix_parts)
    per_comp = np.concatenate(per_parts)
    true = np.concatenate(true_parts)
    sel = pinned_test_mask(dm, dataset)
    mix, per_comp, true = mix[sel], per_comp[sel], true[sel]

    table = compartment_volume_fractions()
    comp_rows: list[CompartmentAgreement] = []
    mix_rows: list[MixtureAgreement] = []
    for k, aa in enumerate(AA_KEYS):
        entries = described[aa]
        best_name, best_rho = "", -np.inf
        # Share of the whole-cell average carried by each compartment, at the median
        # genotype: the weights are assumed, so what they buy has to be visible.
        contributions = np.stack(
            [
                table.fractions[c] * np.power(10.0, per_comp[:, k, j])
                for j, (c, _, _) in enumerate(entries)
            ],
            axis=1,
        )
        total = contributions.sum(axis=1)
        for j, (compartment, sid, weight) in enumerate(entries):
            rho = _masked_spearman(per_comp[:, k, j], true[:, k])
            comp_rows.append(
                CompartmentAgreement(
                    amino_acid=aa,
                    compartment=compartment,
                    species=sid,
                    fractional_volume=weight,
                    n=int(np.isfinite(true[:, k]).sum()),
                    spearman=rho,
                    pearson=_masked_pearson(per_comp[:, k, j], true[:, k]),
                    median_log10_m=float(np.median(per_comp[:, k, j])),
                    median_share_of_mixture=float(
                        np.median(contributions[:, j] / total)
                    ),
                )
            )
            if rho > best_rho:
                best_name, best_rho = compartment, rho
        mix_rows.append(
            MixtureAgreement(
                amino_acid=aa,
                n=int(np.isfinite(true[:, k]).sum()),
                spearman=_masked_spearman(mix[:, k], true[:, k]),
                pearson=_masked_pearson(mix[:, k], true[:, k]),
                median_pred_log10_m=float(np.median(mix[:, k])),
                median_true_log10_m=float(np.nanmedian(true[:, k])),
                best_compartment_by_spearman=best_name,
                best_compartment_spearman=float(best_rho),
            )
        )

    audit = CompartmentAudit(
        result_file=args.result,
        checkpoint=ckpt_path,
        arm=cell.arm,
        seed=cell.seed,
        epochs=cell.epochs,
        select_epoch=int(result["select_epoch"]),
        conc_space=cell.conc_space,
        conc_lower_m=float(layer.config.conc_lower_m),
        conc_upper_m=float(layer.config.conc_upper_m),
        group="pinned_test",
        volume_fractions=table.model_dump(),
        mixture=mix_rows,
        compartments=comp_rows,
        code_provenance=code_provenance(),
    )
    out = args.out or f"conc_compartments_{osp.splitext(args.result)[0]}.json"
    path = osp.join(RESULTS_DIR, out)
    with open(path, "w") as fh:
        json.dump(audit.model_dump(), fh, indent=2)
    print(f"wrote {path}")
    print(
        "\n| amino acid | mixture rho | best compartment | its rho | pred log10 M | measured log10 M |"
    )
    print("|---|---:|---|---:|---:|---:|")
    for row in mix_rows:
        print(
            f"| {row.amino_acid} | {row.spearman:+.4f} | "
            f"{row.best_compartment_by_spearman} | {row.best_compartment_spearman:+.4f} | "
            f"{row.median_pred_log10_m:.2f} | {row.median_true_log10_m:.2f} |"
        )
    print("\n| amino acid | compartment | f_n | share of the average | rho |")
    print("|---|---|---:|---:|---:|")
    for comp_row in comp_rows:
        print(
            f"| {comp_row.amino_acid} | {comp_row.compartment} | "
            f"{comp_row.fractional_volume:.3f} | "
            f"{comp_row.median_share_of_mixture:.3f} | {comp_row.spearman:+.4f} |"
        )


if __name__ == "__main__":
    main()
