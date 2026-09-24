# experiments/032-amino-acid-module/scripts/verify_doc_numbers.py
# [[experiments.032-amino-acid-module.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/verify_doc_numbers

"""Every number typed into notes-tex/032-amino-acid-module's prose, checked against its source file.

The essentiality verifier's rule and helpers, applied to this document: each check names
the claim, the quantity as the document states it, and how to recompute it from the
study's results tree. Add one whenever a number is written into prose. Exits non-zero
on any mismatch. Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/verify_doc_numbers.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "scripts"))
from verify_doc_numbers import (  # noqa: E402
    Check,
    concordance,
    concordance_summary,
    degeneracy,
    ensemble,
    physics_store,
    store_baseline,
    transfer,
    transfer_comparator,
)

RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "032-amino-acid-module", "results")


def polytope(*keys: str):
    """A value of the concentration-polytope store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "target_concentration_polytope_store_amino_acids_mdfcap100poly.json")))
        for k in keys:
            d = d[k]
        return float(d)

    return run


def go_model(arm: str, key: str):
    """A value of one arm of the GO-models read (target_go_models.py)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "go_models_amino_acids.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        return float(a[key])

    return run


CHECKS: list[Check] = [
    # sec:physics -- the annotation-only family on the amino acids.
    Check(claim="go-models: aa GO perceptron", stated=0.054,
          recompute=go_model("go_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa GO perceptron sd", stated=0.002,
          recompute=go_model("go_mlp", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa ProtT5 perceptron", stated=0.054,
          recompute=go_model("prot_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa ProtT5 perceptron sd", stated=0.008,
          recompute=go_model("prot_mlp", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa GO with ProtT5", stated=0.065,
          recompute=go_model("go_prot_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa GO with ProtT5 sd", stated=0.003,
          recompute=go_model("go_prot_mlp", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa GO with ProtT5 ensemble", stated=0.069,
          recompute=go_model("go_prot_mlp", "ensemble_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa DCell hierarchy", stated=0.035,
          recompute=go_model("dcell_style_shared", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa DCell hierarchy sd", stated=0.015,
          recompute=go_model("dcell_style_shared", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa store perceptron", stated=0.019,
          recompute=go_model("store_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa GO with store", stated=0.042,
          recompute=go_model("go_store_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: aa all three", stated=0.045,
          recompute=go_model("go_prot_store_mlp", "single_mean_spearman"), tolerance=0.0006),
    # sec:rounds -- round 138, the linear MOMA store on the amino acids.
    Check(claim="target-transfer: aa MOMA store arm", stated=0.052,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store arm sd", stated=0.006,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store vs control", stated=0.003,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store vs control sd", stated=0.011,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store vs control seeds above", stated=4,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa MOMA store vs parsimonious store", stated=0.012,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store vs parsimonious store sd", stated=0.008,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa MOMA store vs parsimonious store seeds above", stated=5,
          recompute=transfer("amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa MOMA store six-seed ensemble", stated=0.056,
          recompute=ensemble("w155_aadelmoma_h12l1p1_e200", "spearman"), tolerance=0.0006),
    # sec:rounds -- round 141, the driving-force concentration store on the amino acids.
    Check(claim="target-transfer: aa driving-force store arm", stated=0.045,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store arm sd", stated=0.005,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store vs control", stated=-0.005,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store vs control sd", stated=0.007,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store vs control seeds above", stated=2,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa driving-force store vs parsimonious store", stated=0.004,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store vs parsimonious store sd", stated=0.009,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa driving-force store vs parsimonious store seeds above", stated=5,
          recompute=transfer("amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa driving-force store six-seed ensemble", stated=0.050,
          recompute=ensemble("w158_aadelmdf_h12l1p1_e200", "spearman"), tolerance=0.0006),
    # sec:physics -- the sampled cone on the prototroph.
    Check(claim="store-baselines: aa sampled cone ridge", stated=-0.004,
          recompute=store_baseline("amino_acids", "sampled_mean", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa sampled cone kNN", stated=0.031,
          recompute=store_baseline("amino_acids", "sampled_mean", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa sampled cone forest", stated=0.023,
          recompute=store_baseline("amino_acids", "sampled_mean", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="physics: aa sampled cone lethal deletions", stated=161,
          recompute=lambda: float(__import__("json").load(open(osp.join(RESULTS, "target_sampled_flux_store_amino_acids.json")))["n_lethal"]), tolerance=0.5),
    Check(claim="physics: aa sampled cone distance to point", stated=5.9,
          recompute=lambda: float(__import__("json").load(open(osp.join(RESULTS, "target_sampled_flux_store_amino_acids.json")))["median_rel_l2_log1p_mean_vs_source"]), tolerance=0.06),
    # sec:physics -- the driving-force polytope sampled on the prototroph.
    Check(claim="physics: polytope wild-type constrained reactions", stated=249,
          recompute=polytope("wild_type", "n_active"), tolerance=0.5),
    Check(claim="physics: polytope wild-type binding reactions", stated=59,
          recompute=polytope("wild_type", "n_binding"), tolerance=0.5),
    Check(claim="physics: polytope wild-type constrained metabolites", stated=319,
          recompute=polytope("wild_type", "n_constrained_metabolites"), tolerance=0.5),
    Check(claim="physics: polytope wild-type spread over constrained", stated=2.08,
          recompute=polytope("wild_type", "median_sd_ln_c_constrained"), tolerance=0.006),
    Check(claim="physics: polytope wild-type spread over free", stated=2.62,
          recompute=polytope("wild_type", "median_sd_ln_c_free"), tolerance=0.006),
    Check(claim="physics: polytope rows differing from wild type", stated=405,
          recompute=polytope("n_mean_differs_from_wild_type"), tolerance=0.5),
    Check(claim="store-baselines: aa polytope mean ridge", stated=0.016,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_mean", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa polytope mean kNN", stated=0.011,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_mean", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa polytope mean forest", stated=0.008,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_mean", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa polytope spread ridge", stated=0.018,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_spread", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa polytope spread kNN", stated=0.009,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_spread", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa polytope spread forest", stated=0.001,
          recompute=store_baseline("amino_acids", "mdfcap100_poly_spread", "forest", "mean_spearman"), tolerance=0.0006),
    # sec:target-transfer -- rounds 117 and 118.
    Check(
        claim="target-transfer: amino-acid store arm mean",
        stated=0.025,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid store arm sd",
        stated=0.015,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid control mean",
        stated=0.034,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid control sd",
        stated=0.025,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid paired difference",
        stated=-0.009,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid paired sd",
        stated=0.018,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: whole-vector store affine against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: amino-acid control six-seed ensemble",
        stated=0.058,
        recompute=ensemble("w135_aactx_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: amino-acid store six-seed ensemble",
        stated=0.035,
        recompute=ensemble("w135_aastore_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid best comparator",
        stated=0.068,
        recompute=transfer_comparator("mulleder19", "027 concentration readout"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid context comparator",
        stated=0.061,
        recompute=transfer_comparator("mulleder19", "027 context-only"),
    ),
    Check(
        claim="target-transfer: 027 amino-acid pooled comparator",
        stated=0.048,
        recompute=transfer_comparator("mulleder19", "027 pooled"),
    ),
    # sec:target-transfer -- round 124, the amino-acid deletion-only graph.
    Check(
        claim="target-transfer: amino-acid deletion-only arm mean",
        stated=0.049,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only arm sd",
        stated=0.007,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only paired difference",
        stated=0.015,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only paired sd",
        stated=0.026,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only seeds above",
        stated=5,
        recompute=transfer(
            "amino acids: deletion-only cell graphs against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: amino-acid deletion-only six-seed ensemble",
        stated=0.062,
        recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 121, the amino-acid production store.
    Check(
        claim="target-transfer: aa 20-column arm mean",
        stated=0.031,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column arm sd",
        stated=0.004,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column paired difference",
        stated=-0.003,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column paired sd",
        stated=0.025,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 20-column seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: 19 synthesis ratios and the growth ratio against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa 42-summary arm mean",
        stated=0.023,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary arm sd",
        stated=0.009,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary paired difference",
        stated=-0.011,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary paired sd",
        stated=0.027,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa 42-summary seeds above",
        stated=2,
        recompute=transfer(
            "amino acids: growth vector with all 42 summaries against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa 20-column six-seed ensemble",
        stated=0.030,
        recompute=ensemble("w138_aaprodstat_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: aa 42-summary six-seed ensemble",
        stated=0.026,
        recompute=ensemble("w138_aaprodfull_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 126, the amino-acid store on the corrected graph.
    Check(
        claim="target-transfer: aa corrected-graph store arm mean",
        stated=0.040,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store arm sd",
        stated=0.005,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control",
        stated=-0.009,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control sd",
        stated=0.011,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs corrected control seeds above",
        stated=1,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store",
        stated=0.015,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store sd",
        stated=0.014,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: aa corrected-graph store vs old-graph store seeds above",
        stated=6,
        recompute=transfer(
            "amino acids: deletion-only graphs with the store affine against the store affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: aa corrected-graph store six-seed ensemble",
        stated=0.043,
        recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 136, rule counts on the amino acids.
    Check(claim="target-transfer: aa rule-count store arm", stated=0.033,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store arm sd", stated=0.009,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store vs control", stated=-0.016,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store vs control sd", stated=0.012,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store vs control seeds above", stated=1,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa rule-count store vs plain store", stated=-0.007,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against the store affine there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store vs plain store sd", stated=0.013,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against the store affine there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store vs plain store seeds above", stated=2,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against the store affine there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa plain store level", stated=0.04,
          recompute=transfer("amino acids: vector with rule counts on deletion-only graphs against the store affine there", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: aa rule-count store six-seed ensemble", stated=0.037,
          recompute=ensemble("w150_aadelgpr_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: aa plain store six-seed ensemble", stated=0.043,
          recompute=ensemble("w143_aadelstore_h12l1p1_e200", "spearman"), tolerance=0.0006),
    # sec:target-transfer -- round 137, the solve head on the amino acids.
    Check(claim="target-transfer: aa solve head arm", stated=0.053,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head arm sd", stated=0.008,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head vs control", stated=0.004,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head vs control sd", stated=0.011,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head vs control seeds above", stated=4,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa solve head vs store affine", stated=0.013,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head vs store affine sd", stated=0.008,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa solve head vs store affine seeds above", stated=6,
          recompute=transfer("amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa solve head six-seed ensemble", stated=0.063,
          recompute=ensemble("w151_aadelhead256_h12l1p1_e200", "spearman"), tolerance=0.0006),
    # sec:target-transfer -- round 129, the 300-epoch amino-acid control.
    Check(claim="target-transfer: aa 300-epoch control", stated=0.049,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa 300-epoch control sd", stated=0.009,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa 200-epoch control", stated=0.049,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: aa 200-epoch control sd", stated=0.007,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "sd_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: aa 300 vs 200 epochs", stated=-0.001,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa 300 vs 200 epochs sd", stated=0.003,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa 300 vs 200 epochs seeds above", stated=1,
          recompute=transfer("amino acids: deletion-only graphs for 300 epochs against 200", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa 300-epoch six-seed ensemble", stated=0.060,
          recompute=ensemble("w146_aadel_h12l1p1_e300", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: aa 200-epoch six-seed ensemble", stated=0.062,
          recompute=ensemble("w141_aadel_h12l1p1_e200", "spearman"), tolerance=0.0006),
    # sec:target-transfer -- round 122, the wide amino-acid control.
    Check(claim="target-transfer: aa wide control arm", stated=0.034,
          recompute=transfer("amino acids: width 90, two blocks, against the width-12 control", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa wide control arm sd", stated=0.010,
          recompute=transfer("amino acids: width 90, two blocks, against the width-12 control", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: aa wide control vs width-12 control", stated=0.000,
          recompute=transfer("amino acids: width 90, two blocks, against the width-12 control", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa wide control vs width-12 control sd", stated=0.031,
          recompute=transfer("amino acids: width 90, two blocks, against the width-12 control", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: aa wide control seeds above", stated=3,
          recompute=transfer("amino acids: width 90, two blocks, against the width-12 control", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: aa wide control six-seed ensemble", stated=0.050,
          recompute=ensemble("w139_aawide_h90l2p1_e200", "spearman"), tolerance=0.0006),
    # sec:target-physics -- the direction rule, the MOMA and HiGHS stores, degeneracy.
    Check(claim="target-physics: aa sign10 conflicts", stated=84,
          recompute=physics_store("amino_acids", "sign10", "thermo", "n_conflicts_with_plain_wild_type_flux"), tolerance=0.5),
    Check(claim="target-physics: aa sign30 conflicts", stated=44,
          recompute=physics_store("amino_acids", "sign30", "thermo", "n_conflicts_with_plain_wild_type_flux"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA lethal", stated=175,
          recompute=physics_store("amino_acids", "moma", "n_lethal"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA lethal in the source", stated=161,
          recompute=physics_store("amino_acids", "moma", "n_lethal_source"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA ratios moved", stated=63,
          recompute=physics_store("amino_acids", "moma", "n_genes_ratio_moved_gt_0.01"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA vectors moved", stated=567,
          recompute=physics_store("amino_acids", "moma", "n_genes_log1p_vector_moved"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA median distance", stated=0.0,
          recompute=physics_store("amino_acids", "moma", "median_relative_l2_log1p_to_source"), tolerance=0.0005),
    Check(claim="target-physics: SM store no-effect vectors differing", stated=324,
          recompute=degeneracy("SM_gem", "n_no_effect_vector_differs"), tolerance=0.5),
    Check(claim="target-physics: SM store no-effect deletions", stated=918,
          recompute=degeneracy("SM_gem", "n_no_effect"), tolerance=0.5),
    Check(claim="target-physics: aa MOMA store no-effect vectors differing", stated=11,
          recompute=degeneracy("SM_gem_moma", "n_no_effect_vector_differs"), tolerance=0.5),
    # sec:target-physics -- the store read without the backbone.
    Check(claim="store-baselines: aa HiGHS store kNN", stated=0.022,
          recompute=store_baseline("amino_acids", "pfba_highs", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa ProtT5 forest", stated=0.052,
          recompute=store_baseline("amino_acids", "prot_T5_all", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: aa HiGHS store with ProtT5 forest", stated=0.058,
          recompute=store_baseline("amino_acids", "pfba_highs+prot_T5_all", "forest", "mean_spearman"), tolerance=0.0006),
    # sec:target-physics -- Mulleder 2016 against Cooper 2010.
    Check(claim="concordance: shared strains", stated=4051,
          recompute=concordance_summary("n_shared_strains"), tolerance=0.5),
    Check(claim="concordance: alanine Spearman", stated=-0.006,
          recompute=concordance("alanine", "alanine", "spearman"), tolerance=0.0006),
    Check(claim="concordance: glycine Spearman", stated=0.059,
          recompute=concordance("glycine", "glycine", "spearman"), tolerance=0.0006),
    Check(claim="concordance: glycine p", stated=0.0002,
          recompute=concordance("glycine", "glycine", "spearman_p"), tolerance=0.00005),
    Check(claim="concordance: glutamate Spearman", stated=0.043,
          recompute=concordance("glutamate", "glutamate", "spearman"), tolerance=0.0006),
    Check(claim="concordance: aspartate Spearman", stated=0.031,
          recompute=concordance("aspartate", "aspartate", "spearman"), tolerance=0.0006),
    Check(claim="concordance: lysine b top-5 overlap", stated=22,
          recompute=concordance("lysine_b", "lysine", "top5_overlap"), tolerance=0.5),
    Check(claim="concordance: lysine b top-5 expected", stated=10.1,
          recompute=concordance("lysine_b", "lysine", "top5_expected"), tolerance=0.06),
    Check(claim="concordance: lysine b top-5 enrichment", stated=2.2,
          recompute=concordance("lysine_b", "lysine", "top5_enrichment"), tolerance=0.06),
    Check(claim="concordance: matrix median |rho|", stated=0.018,
          recompute=concordance_summary("matrix_abs_median"), tolerance=0.0006),
    Check(claim="concordance: matrix max |rho|", stated=0.127,
          recompute=concordance_summary("matrix_abs_max"), tolerance=0.0006),
    Check(claim="concordance: Mulleder retest Spearman mean", stated=0.20,
          recompute=concordance_summary("mulleder_retest_spearman_mean"), tolerance=0.005),
    Check(claim="concordance: Mulleder retest Pearson mean", stated=0.43,
          recompute=concordance_summary("mulleder_retest_pearson_mean"), tolerance=0.005),
    Check(claim="concordance: Cooper duplicate spread peaks", stated=17,
          recompute=concordance_summary("cooper_spread_n_peaks"), tolerance=0.5),
    Check(claim="concordance: Cooper duplicate rows sd min", stated=0.1,
          recompute=concordance_summary("cooper_spread_sd_dup_min"), tolerance=0.006),
    Check(claim="concordance: Cooper duplicate rows sd max", stated=0.53,
          recompute=concordance_summary("cooper_spread_sd_dup_max"), tolerance=0.006),
    Check(claim="concordance: Cooper all rows sd min", stated=0.11,
          recompute=concordance_summary("cooper_spread_sd_all_min"), tolerance=0.006),
    Check(claim="concordance: Cooper all rows sd max", stated=0.51,
          recompute=concordance_summary("cooper_spread_sd_all_max"), tolerance=0.006),
    Check(claim="concordance: Cooper duplicate rows sd median", stated=0.29,
          recompute=concordance_summary("cooper_spread_sd_dup_median"), tolerance=0.006),
    Check(claim="concordance: Cooper all rows sd median", stated=0.28,
          recompute=concordance_summary("cooper_spread_sd_all_median"), tolerance=0.006),
    Check(claim="concordance: Cooper duplicate Spearman mean", stated=0.06,
          recompute=concordance_summary("cooper_retest_spearman_mean"), tolerance=0.005),
    Check(claim="concordance: Cooper duplicate log-Pearson mean", stated=0.10,
          recompute=concordance_summary("cooper_retest_pearson_log_mean"), tolerance=0.005),
]


def main() -> None:
    failed = 0
    for check in CHECKS:
        got = check.recompute()
        agrees = abs(got - check.stated) < check.tolerance
        failed += 0 if agrees else 1
        mark = "ok  " if agrees else "WRONG"
        print(f"{mark} {check.claim:58s} document {check.stated:,g}  files {got:,.4f}")
    print(f"\n{len(CHECKS)} checks, {failed} disagreeing with the document")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
