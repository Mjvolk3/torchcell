# experiments/031-betaxanthin-module/scripts/verify_doc_numbers.py
# [[experiments.031-betaxanthin-module.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/verify_doc_numbers

"""Every number typed into notes-tex/031-betaxanthin-module's prose, checked against its source file.

The essentiality verifier's rule and helpers, applied to this document: each check names
the claim, the quantity as the document states it, and how to recompute it from the
study's results tree. Add one whenever a number is written into prose. Exits non-zero
on any mismatch. Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/verify_doc_numbers.py
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

RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "031-betaxanthin-module", "results")


def driving_force(variant: str, *keys: str):
    """A value of the driving-force store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"target_driving_force_store_betaxanthin_{variant}.json")))
        for k in keys:
            d = d[k]
        return float(d)

    return run


def polytope(*keys: str):
    """A value of the concentration-polytope store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "target_concentration_polytope_store_betaxanthin_mdfcap100poly.json")))
        for k in keys:
            d = d[k]
        return float(d)

    return run


def go_model(arm: str, key: str):
    """A value of one arm of the GO-models read (target_go_models.py)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "go_models_betaxanthin.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "n_columns_with_train_support":
            return float(d["n_columns_with_train_support"])
        if key == "select_epoch_min":
            return float(min(r["select_epoch"] for r in a["seeds"]))
        if key == "select_epoch_max":
            return float(max(r["select_epoch"] for r in a["seeds"]))
        return float(a[key])

    return run


CHECKS: list[Check] = [
    # sec:physics -- the annotation-only family.
    Check(claim="go-models: GO columns with train support", stated=2341,
          recompute=go_model("go_mlp", "n_columns_with_train_support"), tolerance=0.5),
    Check(claim="go-models: GO perceptron", stated=0.200,
          recompute=go_model("go_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: GO perceptron sd", stated=0.002,
          recompute=go_model("go_mlp", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: GO perceptron top-50", stated=4.79,
          recompute=go_model("go_mlp", "single_mean_top50"), tolerance=0.006),
    Check(claim="go-models: GO perceptron earliest selected epoch", stated=21,
          recompute=go_model("go_mlp", "select_epoch_min"), tolerance=0.5),
    Check(claim="go-models: GO perceptron latest selected epoch", stated=22,
          recompute=go_model("go_mlp", "select_epoch_max"), tolerance=0.5),
    Check(claim="go-models: store perceptron", stated=0.119,
          recompute=go_model("store_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: store perceptron top-50", stated=2.00,
          recompute=go_model("store_mlp", "single_mean_top50"), tolerance=0.006),
    Check(claim="go-models: ProtT5 perceptron", stated=0.103,
          recompute=go_model("prot_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: ProtT5 perceptron top-50", stated=4.13,
          recompute=go_model("prot_mlp", "single_mean_top50"), tolerance=0.006),
    Check(claim="go-models: GO with store", stated=0.198,
          recompute=go_model("go_store_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: GO with store sd", stated=0.041,
          recompute=go_model("go_store_mlp", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: GO with ProtT5", stated=0.173,
          recompute=go_model("go_prot_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: GO with ProtT5 top-50", stated=4.99,
          recompute=go_model("go_prot_mlp", "single_mean_top50"), tolerance=0.006),
    Check(claim="go-models: GO with ProtT5 ensemble top-50", stated=5.39,
          recompute=go_model("go_prot_mlp", "ensemble_top50"), tolerance=0.006),
    Check(claim="go-models: all three", stated=0.175,
          recompute=go_model("go_prot_store_mlp", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: all three top-50", stated=5.03,
          recompute=go_model("go_prot_store_mlp", "single_mean_top50"), tolerance=0.006),
    Check(claim="go-models: DCell hierarchy", stated=0.049,
          recompute=go_model("dcell_style_shared", "single_mean_spearman"), tolerance=0.0006),
    Check(claim="go-models: DCell hierarchy sd", stated=0.043,
          recompute=go_model("dcell_style_shared", "single_sd_spearman"), tolerance=0.0006),
    Check(claim="go-models: DCell earliest selected epoch", stated=8,
          recompute=go_model("dcell_style_shared", "select_epoch_min"), tolerance=0.5),
    Check(claim="go-models: DCell latest selected epoch", stated=13,
          recompute=go_model("dcell_style_shared", "select_epoch_max"), tolerance=0.5),
    # sec:rounds -- round 132, the wide control on the deletion-only graph.
    Check(claim="target-transfer: wide deletion-only control", stated=0.092,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: wide deletion-only control sd", stated=0.043,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: wide deletion-only control vs old-graph wide control", stated=-0.017,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide deletion-only control vs old-graph wide control sd", stated=0.08,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide deletion-only control vs old-graph wide control seeds above", stated=2,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: wide deletion-only control top-50", stated=4.49,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: old-graph wide control top-50", stated=4.73,
          recompute=transfer("betaxanthin: width 90 on deletion-only graphs against the width-90 control", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: wide store vs wide control on the deletion-only graph", stated=0.028,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide store vs wide control on the deletion-only graph sd", stated=0.037,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide store vs wide control on the deletion-only graph seeds above", stated=4,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: wide store on the deletion-only graph top-50", stated=4.43,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: wide deletion-only control six-seed ensemble", stated=0.122,
          recompute=ensemble("w149_bxwidedel_h90l2p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: wide deletion-only control six-seed ensemble top-50", stated=5.19,
          recompute=ensemble("w149_bxwidedel_h90l2p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 134, the solve head on betaxanthin.
    Check(claim="target-transfer: solve head arm", stated=0.07,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: solve head arm sd", stated=0.049,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: solve head vs control", stated=0.012,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: solve head vs control sd", stated=0.074,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: solve head vs control seeds above", stated=3,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: solve head vs store affine", stated=-0.051,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: solve head vs store affine sd", stated=0.061,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: solve head vs store affine seeds above", stated=1,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: solve head top-50", stated=3.89,
          recompute=transfer("betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: solve head six-seed ensemble", stated=0.095,
          recompute=ensemble("w153_bxdelhead256_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: solve head six-seed ensemble top-50", stated=5.39,
          recompute=ensemble("w153_bxdelhead256_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 135, the linear MOMA store on betaxanthin.
    Check(claim="target-transfer: MOMA store arm", stated=0.107,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store arm sd", stated=0.018,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store vs control", stated=0.049,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store vs control sd", stated=0.064,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store vs control seeds above", stated=5,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: MOMA store vs parsimonious store", stated=-0.014,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store vs parsimonious store sd", stated=0.029,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store vs parsimonious store seeds above", stated=1,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: MOMA store top-50", stated=4.99,
          recompute=transfer("betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: MOMA store six-seed ensemble", stated=0.115,
          recompute=ensemble("w154_bxdelmoma_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA store six-seed ensemble top-50", stated=5.79,
          recompute=ensemble("w154_bxdelmoma_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 133, the rule counts on betaxanthin.
    Check(claim="target-transfer: rule-count store arm", stated=0.114,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store arm sd", stated=0.019,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store vs control", stated=0.056,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store vs control sd", stated=0.056,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store vs control seeds above", stated=5,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: rule-count store vs plain store", stated=-0.007,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against the store affine there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store vs plain store sd", stated=0.023,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against the store affine there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store vs plain store seeds above", stated=1,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against the store affine there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: rule-count store top-50", stated=3.96,
          recompute=transfer("betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: rule-count store six-seed ensemble", stated=0.137,
          recompute=ensemble("w152_bxdelgpr_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: rule-count store six-seed ensemble top-50", stated=4.79,
          recompute=ensemble("w152_bxdelgpr_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 139, the HiGHS parsimonious store on betaxanthin.
    Check(claim="target-transfer: HiGHS store arm", stated=0.13,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store arm sd", stated=0.034,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store vs control", stated=0.072,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store vs control sd", stated=0.067,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store vs control seeds above", stated=5,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: HiGHS store vs GLPK store", stated=0.009,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store vs GLPK store sd", stated=0.041,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store vs GLPK store seeds above", stated=5,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: HiGHS store top-50", stated=4.46,
          recompute=transfer("betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: HiGHS store six-seed ensemble", stated=0.148,
          recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store six-seed ensemble top-50", stated=5.79,
          recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 140, the driving-force concentration store on betaxanthin.
    Check(claim="target-transfer: driving-force store arm", stated=0.118,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store arm sd", stated=0.055,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store vs control", stated=0.059,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store vs control sd", stated=0.055,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store vs control seeds above", stated=5,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: driving-force store vs GLPK store", stated=-0.003,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store vs GLPK store sd", stated=0.037,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store vs GLPK store seeds above", stated=2,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: driving-force store top-50", stated=4.06,
          recompute=transfer("betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: driving-force store six-seed ensemble", stated=0.133,
          recompute=ensemble("w157_bxdelmdf_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: driving-force store six-seed ensemble top-50", stated=4.99,
          recompute=ensemble("w157_bxdelmdf_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 142, the sampled cone on betaxanthin.
    Check(claim="target-transfer: sampled cone arm", stated=0.088,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone arm sd", stated=0.022,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone vs control", stated=0.03,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone vs control sd", stated=0.07,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone vs control seeds above", stated=4,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: sampled cone vs GLPK store", stated=-0.033,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone vs GLPK store sd", stated=0.024,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone vs GLPK store seeds above", stated=0,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: sampled cone top-50", stated=2.9,
          recompute=transfer("betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: sampled cone six-seed ensemble", stated=0.083,
          recompute=ensemble("w159_bxdelsampled_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: sampled cone six-seed ensemble top-50", stated=3.00,
          recompute=ensemble("w159_bxdelsampled_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 144, the store arm at width 6.
    Check(claim="target-transfer: width-6 store arm", stated=0.113,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: width-6 store arm sd", stated=0.038,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: width-6 store vs width-12 store", stated=-0.008,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: width-6 store vs width-12 store sd", stated=0.055,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: width-6 store vs width-12 store seeds above", stated=4,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: width-6 store top-50", stated=4.19,
          recompute=transfer("betaxanthin: deletion-only store at width 6 against width 12", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: width-6 store six-seed ensemble", stated=0.129,
          recompute=ensemble("w161_bxdelstoreh6_h6l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: width-6 store six-seed ensemble top-50", stated=5.39,
          recompute=ensemble("w161_bxdelstoreh6_h6l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 145, the store arm at dropout 0.3.
    Check(claim="target-transfer: dropout-0.3 store arm", stated=0.145,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: dropout-0.3 store arm sd", stated=0.018,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: dropout-0.3 store vs dropout-0.1 store", stated=0.024,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: dropout-0.3 store vs dropout-0.1 store sd", stated=0.04,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: dropout-0.3 store vs dropout-0.1 store seeds above", stated=4,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: dropout-0.3 store top-50", stated=5.03,
          recompute=transfer("betaxanthin: deletion-only store at dropout 0.3 against 0.1", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: dropout-0.3 store six-seed ensemble", stated=0.149,
          recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: dropout-0.3 store six-seed ensemble top-50", stated=5.79,
          recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 147, the polytope mean on betaxanthin.
    Check(claim="target-transfer: polytope mean arm", stated=0.125,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean arm sd", stated=0.041,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs control", stated=0.067,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs control sd", stated=0.047,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs control seeds above", stated=6,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: polytope mean vs GLPK store", stated=0.004,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs GLPK store sd", stated=0.047,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs GLPK store seeds above", stated=4,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: polytope mean vs driving-force point", stated=0.008,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs driving-force point sd", stated=0.054,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean vs driving-force point seeds above", stated=3,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: polytope mean top-50", stated=3.69,
          recompute=transfer("betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: polytope mean six-seed ensemble", stated=0.151,
          recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: polytope mean six-seed ensemble top-50", stated=5.59,
          recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 149, the release rule on the MOMA store.
    Check(claim="target-transfer: MOMA release arm", stated=0.106,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release arm sd", stated=0.034,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release vs MOMA store", stated=-0.001,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release vs MOMA store sd", stated=0.038,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release vs MOMA store seeds above", stated=3,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: MOMA release vs GLPK release", stated=-0.004,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release vs GLPK release sd", stated=0.063,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release vs GLPK release seeds above", stated=2,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: MOMA release top-50", stated=4.53,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the MOMA store", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: GLPK release level", stated=0.11,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: GLPK release top-50", stated=4.66,
          recompute=transfer("betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: MOMA release six-seed ensemble", stated=0.105,
          recompute=ensemble("w166_bxdelmomarel_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: MOMA release six-seed ensemble top-50", stated=5.59,
          recompute=ensemble("w166_bxdelmomarel_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 150, GO in the affine slot on betaxanthin.
    Check(claim="target-transfer: GO affine arm", stated=0.202,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine arm sd", stated=0.001,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine vs control", stated=0.144,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine vs control sd", stated=0.075,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine vs control seeds above", stated=6,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: GO affine vs GLPK store", stated=0.082,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine vs GLPK store sd", stated=0.025,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine vs GLPK store seeds above", stated=6,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: GO affine top-50", stated=5.03,
          recompute=transfer("betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: GO affine six-seed ensemble", stated=0.202,
          recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: GO affine six-seed ensemble top-50", stated=4.99,
          recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 151, the HiGHS store with GO appended.
    Check(claim="target-transfer: GO with HiGHS arm", stated=0.211,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS arm sd", stated=0.007,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS vs HiGHS alone", stated=0.081,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS vs HiGHS alone sd", stated=0.032,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS vs HiGHS alone seeds above", stated=6,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: GO with HiGHS vs GO alone", stated=0.009,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against GO alone", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS vs GO alone sd", stated=0.007,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against GO alone", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS vs GO alone seeds above", stated=5,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against GO alone", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: GO with HiGHS top-50", stated=4.83,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: HiGHS alone top-50 (round 151 pair)", stated=4.46,
          recompute=transfer("betaxanthin: HiGHS store with GO appended against the HiGHS store alone", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: GO with HiGHS six-seed ensemble", stated=0.212,
          recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: GO with HiGHS six-seed ensemble top-50", stated=4.99,
          recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:rounds -- round 152, the release rule on the HiGHS store.
    Check(claim="target-transfer: HiGHS release arm", stated=0.108,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release arm sd", stated=0.021,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs HiGHS store", stated=-0.022,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs HiGHS store sd", stated=0.042,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs HiGHS store seeds above", stated=2,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: HiGHS store level (round 152 pair)", stated=0.13,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs GLPK release", stated=-0.002,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs GLPK release sd", stated=0.058,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release vs GLPK release seeds above", stated=2,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: HiGHS release top-50", stated=3.99,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: HiGHS store top-50 (round 152 pair)", stated=4.46,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the HiGHS store", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: GLPK release top-50 (round 152 pair)", stated=4.66,
          recompute=transfer("betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: HiGHS release six-seed ensemble", stated=0.139,
          recompute=ensemble("w169_bxdelpfbarel_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS release six-seed ensemble top-50", stated=4.99,
          recompute=ensemble("w169_bxdelpfbarel_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    Check(claim="target-transfer: HiGHS store six-seed ensemble (round 152 pair)", stated=0.148,
          recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: HiGHS store six-seed ensemble top-50 (round 152 pair)", stated=5.79,
          recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    Check(claim="target-transfer: GLPK release six-seed ensemble (round 152 pair)", stated=0.12,
          recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: GLPK release six-seed ensemble top-50 (round 152 pair)", stated=6.19,
          recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:physics -- the driving-force polytope sampled.
    Check(claim="physics: polytope wild-type constrained metabolites", stated=234,
          recompute=polytope("wild_type", "n_constrained_metabolites"), tolerance=0.5),
    Check(claim="physics: polytope wild-type spread over constrained", stated=2.18,
          recompute=polytope("wild_type", "median_sd_ln_c_constrained"), tolerance=0.006),
    Check(claim="physics: polytope wild-type spread over free", stated=2.63,
          recompute=polytope("wild_type", "median_sd_ln_c_free"), tolerance=0.006),
    Check(claim="physics: polytope rows differing from wild type", stated=455,
          recompute=polytope("n_mean_differs_from_wild_type"), tolerance=0.5),
    Check(claim="physics: polytope viable rows solved", stated=1062,
          recompute=polytope("n_solved"), tolerance=0.5),
    Check(claim="physics: polytope mean distance to point", stated=0.12,
          recompute=polytope("median_rel_distance_mean_to_point_viable"), tolerance=0.006),
    Check(claim="store-baselines: polytope mean ridge", stated=0.136,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_mean", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: polytope mean kNN", stated=0.127,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_mean", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: polytope mean forest", stated=0.126,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_mean", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: polytope spread ridge", stated=0.096,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_spread", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: polytope spread kNN", stated=0.138,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_spread", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: polytope spread forest", stated=0.128,
          recompute=store_baseline("betaxanthin", "mdfcap100_poly_spread", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: capped concentrations kNN", stated=0.072,
          recompute=store_baseline("betaxanthin", "mdfcap100_conc", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: capped concentrations forest", stated=0.098,
          recompute=store_baseline("betaxanthin", "mdfcap100_conc", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: HiGHS store forest", stated=0.117,
          recompute=store_baseline("betaxanthin", "pfba_highs", "forest", "mean_spearman"), tolerance=0.0006),
    # sec:physics -- the max-min driving-force store.
    Check(claim="physics: driving-force wild-type optimum", stated=-1.16,
          recompute=driving_force("mdfcap100", "wild_type", "mdf_kj_per_mol"), tolerance=0.006),
    Check(claim="physics: driving-force wild-type constrained reactions", stated=172,
          recompute=driving_force("mdfcap100", "wild_type", "n_active"), tolerance=0.5),
    Check(claim="physics: driving-force wild-type binding reactions", stated=32,
          recompute=driving_force("mdfcap100", "wild_type", "n_binding"), tolerance=0.5),
    Check(claim="physics: driving-force viable deletions solved", stated=1062,
          recompute=driving_force("mdfcap100", "n_solved"), tolerance=0.5),
    Check(claim="physics: driving-force deletions with a negative optimum", stated=1062,
          recompute=driving_force("mdfcap100", "n_mdf_negative"), tolerance=0.5),
    Check(claim="physics: driving-force deletions moving the concentrations", stated=451,
          recompute=driving_force("mdfcap100", "n_conc_moved"), tolerance=0.5),
    # sec:target-transfer -- rounds 117 and 118.
    Check(
        claim="target-transfer: betaxanthin store arm mean",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin store arm sd",
        stated=0.011,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin control mean",
        stated=0.079,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin control sd",
        stated=0.056,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin paired difference",
        stated=0.043,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin paired sd",
        stated=0.049,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: betaxanthin store top-50 enrichment",
        stated=3.49,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin control top-50 enrichment",
        stated=4.06,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin store six-seed ensemble",
        stated=0.135,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: betaxanthin store triple 0",
        stated=0.146,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman", "triple0"),
    ),
    Check(
        claim="target-transfer: betaxanthin store triple 1",
        stated=0.133,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman", "triple1"),
    ),
    Check(
        claim="target-transfer: betaxanthin control six-seed ensemble",
        stated=0.076,
        recompute=ensemble("w134_bxctx_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: betaxanthin store ensemble top-50",
        stated=4.39,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin control ensemble top-50",
        stated=4.59,
        recompute=ensemble("w134_bxctx_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: 027 precursor comparator",
        stated=0.131,
        recompute=transfer_comparator("betaxanthin", "027 flux precursor"),
    ),
    Check(
        claim="target-transfer: 027 context comparator",
        stated=0.115,
        recompute=transfer_comparator("betaxanthin", "027 context-only"),
    ),
    Check(
        claim="target-transfer: kNN comparator",
        stated=0.145,
        recompute=transfer_comparator("betaxanthin", "cosine kNN"),
    ),
    # sec:target-transfer -- rounds 119 and 120.
    Check(
        claim="target-transfer: two summary columns arm mean",
        stated=0.101,
        recompute=transfer(
            "betaxanthin: two summary columns against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: two summary columns arm sd",
        stated=0.061,
        recompute=transfer(
            "betaxanthin: two summary columns against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: two summary columns paired difference",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: two summary columns paired sd",
        stated=0.072,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: two summary columns seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: two summary columns single-seed top-50",
        stated=3.03,
        recompute=transfer(
            "betaxanthin: two summary columns against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: two summary columns six-seed ensemble",
        stated=0.071,
        recompute=ensemble("w136_bxratio_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: decayed affine arm mean",
        stated=0.073,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine arm sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine paired difference",
        stated=-0.005,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine paired sd",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only graph arm mean",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph arm sd",
        stated=0.075,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph paired difference",
        stated=-0.020,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph paired sd",
        stated=0.076,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rule-count columns arm mean",
        stated=0.074,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns arm sd",
        stated=0.059,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns paired difference",
        stated=-0.004,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns paired sd",
        stated=0.092,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rounds 119/120 ensemble top-50 floor (deletion-only)",
        stated=4.79,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 ensemble top-50 ceiling (decayed affine)",
        stated=5.19,
        recompute=ensemble("w136_bxstorewd_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 single-seed top-50 floor (rule counts)",
        stated=2.96,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 single-seed top-50 ceiling (decayed affine)",
        stated=3.59,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 123, the deletion-only graph with the store affine.
    Check(
        claim="target-transfer: deletion-only store arm mean",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store arm sd",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, paired",
        stated=0.063,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, paired sd",
        stated=0.065,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, paired",
        stated=0.000,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, paired sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store single-seed top-50",
        stated=4.93,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: deletion-only store ensemble top-50",
        stated=5.59,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: deletion-only store six-seed ensemble",
        stated=0.131,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 122, the width-90 betaxanthin control.
    Check(
        claim="target-transfer: width-90 control arm mean",
        stated=0.109,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control arm sd",
        stated=0.053,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control paired difference",
        stated=0.030,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control paired sd",
        stated=0.079,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: width-90 control single-seed top-50",
        stated=4.73,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: width-90 control six-seed ensemble",
        stated=0.134,
        recompute=ensemble("w139_bxwide_h90l2p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: width-90 control ensemble top-50",
        stated=5.59,
        recompute=ensemble("w139_bxwide_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 123, the store affine at width 90.
    Check(
        claim="target-transfer: wide store arm mean",
        stated=0.134,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: wide store arm sd",
        stated=0.020,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: wide store paired difference",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: wide store paired sd",
        stated=0.051,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: wide store seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: wide store single-seed top-50",
        stated=4.09,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: wide store six-seed ensemble",
        stated=0.153,
        recompute=ensemble("w140_bxwidestore_h90l2p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: wide store ensemble top-50",
        stated=5.19,
        recompute=ensemble("w140_bxwidestore_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 125, the growth and production blocks alone.
    Check(
        claim="target-transfer: growth block arm mean",
        stated=0.033,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: growth block arm sd",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector",
        stated=-0.088,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector sd",
        stated=0.029,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: growth block vs control",
        stated=-0.046,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: growth block vs control sd",
        stated=0.064,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: growth block vs control seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: growth block single-seed top-50",
        stated=3.06,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: growth block six-seed ensemble",
        stated=0.059,
        recompute=ensemble("w142_bxgrowth_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: production block arm mean",
        stated=0.091,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: production block arm sd",
        stated=0.040,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector",
        stated=-0.030,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector sd",
        stated=0.034,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: production block vs control",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: the production block alone against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs control sd",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: the production block alone against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: production block vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: the production block alone against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: production block single-seed top-50",
        stated=3.63,
        recompute=transfer(
            "betaxanthin: the production block alone against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: production block six-seed ensemble",
        stated=0.112,
        recompute=ensemble("w142_bxprod_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 127, twelve seeds of the deletion-only store.
    Check(
        claim="target-transfer: deletion-only store twelve-seed single mean",
        stated=0.116,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "mean_single_spearman", "arm"),
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed count",
        stated=12,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "n_seeds", "arm"),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed ensemble",
        stated=0.129,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "spearman", "all_seeds"),
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed ensemble top-50",
        stated=5.39,
        recompute=ensemble(
            "w140_bxdelstore_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 128, the wide store on the deletion-only graph.
    Check(claim="target-transfer: wide corrected-graph store arm", stated=0.120,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store arm sd", stated=0.024,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store vs wide control", stated=0.011,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store vs wide control sd", stated=0.067,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store vs wide control seeds above", stated=3,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: wide corrected-graph store vs width-12 store", stated=-0.001,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-12 one", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store vs width-12 store sd", stated=0.035,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-12 one", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store top-50", stated=4.43,
          recompute=transfer("betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: wide corrected-graph store six-seed ensemble", stated=0.147,
          recompute=ensemble("w145_bxwidedelstore_h90l2p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: wide corrected-graph store six-seed ensemble top-50", stated=4.99,
          recompute=ensemble("w145_bxwidedelstore_h90l2p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:target-transfer -- rounds 130 and 131, the two affine schedules.
    Check(claim="target-transfer: affine at 40 arm", stated=0.074,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 arm sd", stated=0.046,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 store level", stated=0.121,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs store", stated=-0.047,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs store sd", stated=0.053,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs store seeds above", stated=2,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: affine at 40 control level", stated=0.058,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs", "mean_spearman", "control"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs control", stated=0.016,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs control sd", stated=0.068,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 vs control seeds above", stated=4,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: affine at 40 top-50", stated=3.86,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: affine at 40 store top-50", stated=4.93,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: affine at 40 control top-50", stated=3.39,
          recompute=transfer("betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs", "mean_top50_enrichment", "control"), tolerance=0.006),
    Check(claim="target-transfer: affine at 40 six-seed ensemble", stated=0.129,
          recompute=ensemble("w147_bxdelstore40_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: affine at 40 six-seed ensemble top-50", stated=5.79,
          recompute=ensemble("w147_bxdelstore40_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    Check(claim="target-transfer: affine from escape arm", stated=0.11,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "mean_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape arm sd", stated=0.05,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "sd_spearman", "arm"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape vs store", stated=-0.01,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape vs store sd", stated=0.052,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape vs store seeds above", stated=1,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: affine from escape vs control", stated=0.052,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against deletion-only graphs", "mean_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape vs control sd", stated=0.035,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against deletion-only graphs", "sd_difference"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape vs control seeds above", stated=6,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against deletion-only graphs", "n_positive"), tolerance=0.5),
    Check(claim="target-transfer: affine from escape top-50", stated=4.66,
          recompute=transfer("betaxanthin: deletion-only store, affine from escape, against the deletion-only store", "mean_top50_enrichment", "arm"), tolerance=0.006),
    Check(claim="target-transfer: affine from escape six-seed ensemble", stated=0.120,
          recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "spearman"), tolerance=0.0006),
    Check(claim="target-transfer: affine from escape six-seed ensemble top-50", stated=6.19,
          recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "top50_enrichment"), tolerance=0.006),
    # sec:target-physics -- the direction rule, the MOMA and HiGHS stores, degeneracy.
    Check(claim="target-physics: bx sign10 fixed forward", stated=1754,
          recompute=physics_store("betaxanthin", "sign10", "thermo", "n_fixed_forward"), tolerance=0.5),
    Check(claim="target-physics: bx sign10 fixed reverse", stated=1068,
          recompute=physics_store("betaxanthin", "sign10", "thermo", "n_fixed_reverse"), tolerance=0.5),
    Check(claim="target-physics: bx sign10 conflicts", stated=97,
          recompute=physics_store("betaxanthin", "sign10", "thermo", "n_conflicts_with_plain_wild_type_flux"), tolerance=0.5),
    Check(claim="target-physics: bx sign10 wild type infeasible", stated=1,
          recompute=physics_store("betaxanthin", "sign10", "wild_type_infeasible"), tolerance=0.5),
    Check(claim="target-physics: bx sign30 fixed forward", stated=1071,
          recompute=physics_store("betaxanthin", "sign30", "thermo", "n_fixed_forward"), tolerance=0.5),
    Check(claim="target-physics: bx sign30 fixed reverse", stated=905,
          recompute=physics_store("betaxanthin", "sign30", "thermo", "n_fixed_reverse"), tolerance=0.5),
    Check(claim="target-physics: bx sign30 conflicts", stated=69,
          recompute=physics_store("betaxanthin", "sign30", "thermo", "n_conflicts_with_plain_wild_type_flux"), tolerance=0.5),
    Check(claim="target-physics: bx MOMA lethal", stated=107,
          recompute=physics_store("betaxanthin", "moma", "n_lethal"), tolerance=0.5),
    Check(claim="target-physics: bx MOMA lethal in the source", stated=93,
          recompute=physics_store("betaxanthin", "moma", "n_lethal_source"), tolerance=0.5),
    Check(claim="target-physics: bx MOMA ratios moved", stated=59,
          recompute=physics_store("betaxanthin", "moma", "n_genes_ratio_moved_gt_0.01"), tolerance=0.5),
    Check(claim="target-physics: bx MOMA vectors moved", stated=1056,
          recompute=physics_store("betaxanthin", "moma", "n_genes_log1p_vector_moved"), tolerance=0.5),
    Check(claim="target-physics: bx MOMA median distance", stated=0.39,
          recompute=physics_store("betaxanthin", "moma", "median_relative_l2_log1p_to_source"), tolerance=0.005),
    Check(claim="target-physics: bx HiGHS lethal", stated=93,
          recompute=physics_store("betaxanthin", "pfba", "n_lethal"), tolerance=0.5),
    Check(claim="target-physics: bx HiGHS ratios moved", stated=17,
          recompute=physics_store("betaxanthin", "pfba", "n_genes_ratio_moved_gt_0.01"), tolerance=0.5),
    Check(claim="target-physics: bx HiGHS median distance to GLPK", stated=0.015,
          recompute=physics_store("betaxanthin", "pfba", "median_relative_l2_log1p_to_source"), tolerance=0.0005),
    Check(claim="target-physics: GLPK store no-effect deletions", stated=1031,
          recompute=degeneracy("YPD_bx", "n_no_effect"), tolerance=0.5),
    Check(claim="target-physics: GLPK store no-effect vectors differing", stated=1025,
          recompute=degeneracy("YPD_bx", "n_no_effect_vector_differs"), tolerance=0.5),
    Check(claim="target-physics: GLPK store no-effect median distance", stated=0.38,
          recompute=degeneracy("YPD_bx", "median_rel_l2_log1p"), tolerance=0.005),
    Check(claim="target-physics: HiGHS store no-effect vectors differing", stated=428,
          recompute=degeneracy("YPD_bx_pfba", "n_no_effect_vector_differs"), tolerance=0.5),
    Check(claim="target-physics: HiGHS store no-effect p90 distance", stated=0.39,
          recompute=degeneracy("YPD_bx_pfba", "p90_rel_l2_log1p"), tolerance=0.005),
    Check(claim="target-physics: bx MOMA store no-effect vectors differing", stated=18,
          recompute=degeneracy("YPD_bx_moma", "n_no_effect_vector_differs"), tolerance=0.5),
    # sec:target-physics -- the store read without the backbone.
    Check(claim="store-baselines: GLPK growth block ridge", stated=-0.006,
          recompute=store_baseline("betaxanthin", "pfba_glpk_growth", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: GLPK growth block kNN", stated=-0.004,
          recompute=store_baseline("betaxanthin", "pfba_glpk_growth", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: GLPK growth block forest", stated=0.009,
          recompute=store_baseline("betaxanthin", "pfba_glpk_growth", "forest", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: GLPK whole vector kNN", stated=0.097,
          recompute=store_baseline("betaxanthin", "pfba_glpk_full", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: HiGHS store kNN", stated=0.195,
          recompute=store_baseline("betaxanthin", "pfba_highs", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: HiGHS store kNN top-50", stated=3.39,
          recompute=store_baseline("betaxanthin", "pfba_highs", "knn", "top50_enrichment"), tolerance=0.006),
    Check(claim="store-baselines: MOMA store ridge", stated=0.135,
          recompute=store_baseline("betaxanthin", "moma", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: flux statistics kNN", stated=0.145,
          recompute=store_baseline("betaxanthin", "flux_stats", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: ProtT5 kNN", stated=0.144,
          recompute=store_baseline("betaxanthin", "prot_T5_all", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: ProtT5 kNN top-50", stated=4.59,
          recompute=store_baseline("betaxanthin", "prot_T5_all", "knn", "top50_enrichment"), tolerance=0.006),
    Check(claim="store-baselines: GLPK store with ProtT5 forest top-50", stated=4.99,
          recompute=store_baseline("betaxanthin", "pfba_glpk_full+prot_T5_all", "forest", "top50_enrichment"), tolerance=0.006),
    Check(claim="store-baselines: capped driving-force concentrations ridge", stated=0.152,
          recompute=store_baseline("betaxanthin", "mdfcap100_conc", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: capped driving-force concentrations ridge top-50", stated=3.39,
          recompute=store_baseline("betaxanthin", "mdfcap100_conc", "ridge", "top50_enrichment"), tolerance=0.006),
    Check(claim="store-baselines: sampled-cone mean ridge", stated=0.007,
          recompute=store_baseline("betaxanthin", "sampled_mean", "ridge", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: sampled-cone mean kNN", stated=0.064,
          recompute=store_baseline("betaxanthin", "sampled_mean", "knn", "mean_spearman"), tolerance=0.0006),
    Check(claim="store-baselines: sampled-cone mean forest", stated=0.102,
          recompute=store_baseline("betaxanthin", "sampled_mean", "forest", "mean_spearman"), tolerance=0.0006),
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
