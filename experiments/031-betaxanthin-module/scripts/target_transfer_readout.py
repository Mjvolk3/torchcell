# experiments/031-betaxanthin-module/scripts/target_transfer_readout.py
# [[experiments.031-betaxanthin-module.scripts.target_transfer_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_transfer_readout

r"""The layer-free module carried to betaxanthin and the amino acids: paired reads.

Rounds 117 to 119 train the 028 backbone (pathway multi-hot with ProtT5, width 12) on
the Cachera betaxanthin screen and on Mulleder's 19 amino acids under the 027 protocol
(Merzbacher's 639 genes pinned to test, CRPS, the epoch selected on validation Pearson),
with and without the store affine. Every arm is paired by seed against the control that
shares its backbone and has no affine. The betaxanthin statistic is the pinned-test
Spearman and the top-50 enrichment of high producers; the amino-acid statistic is the
mean per-column Spearman over the 19 amino acids on the pinned genotypes.

The comparators are read from 027's own result files, on the same pinned split and the
same statistic: the betaxanthin precursor readout (wave 8, three seeds) and its
context-only control (wave 9), the cosine kNN over ProtT5 and the FCL RandomForest
(the linear-baseline and Merzbacher comparison files); for the amino acids the
concentration readout without constraints (wave 7), the context-only concentration
control (wave 8), the pooled arm (wave 7) and the bilinear ridge that the linear ladder
selected on validation. Nothing is typed in from prose.

Writes ``transfer_readout.json`` into each study's results tree (031 betaxanthin, 032 amino acids). Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_transfer_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
#: Each target is its own study (2026-09-23): betaxanthin is 031, the amino acids 032.
RESULTS_BY_HEAD = {
    "betaxanthin": osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "results"),
    "mulleder19": osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results"),
}
RESULTS_027 = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
BASE_027 = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "conf", "base.yaml")
SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
SEEDS_027 = (2700, 2701, 2702)


def comparators() -> dict[str, dict[str, float]]:
    """The 027 numbers, each the mean of its seeds' pinned-test statistic."""

    def arm(stem: str, key: str) -> float:
        vals = []
        for s in SEEDS_027:
            with open(osp.join(RESULTS_027, f"{stem}_s{s}.json")) as fh:
                vals.append(float(json.load(fh)["test"]["pinned"][key]))
        return float(np.mean(vals))

    def ladder(target: str, block: str) -> float:
        with open(osp.join(RESULTS_027, f"linear_baselines_{target}.json")) as fh:
            return float(json.load(fh)[block]["headline_test_pinned"]["mean_spearman"])

    with open(BASE_027) as fh:
        fcl = float(yaml.safe_load(fh)["baselines"]["fcl_rf_test_spearman"])
    return {
        "betaxanthin": {
            "027 flux precursor readout (wave 8), three seeds": arm(
                "wave8_bxprec_e200", "spearman"
            ),
            "027 context-only control of the precursor readout (wave 9), three seeds": arm(
                "wave9_bxctxprec_e200", "spearman"
            ),
            "cosine kNN over ProtT5 of the deleted gene (k 25, job 1909)": ladder(
                "betaxanthin", "B3_neighbor_average"
            ),
            "FCL RandomForest (Merzbacher 2025)": fcl,
        },
        "mulleder19": {
            "027 concentration readout without constraints (wave 7), three seeds": arm(
                "wave7_aa_conc_nocstr_e200", "mean_spearman"
            ),
            "027 context-only concentration control (wave 8), three seeds": arm(
                "wave8_aa_ctxconc_e200", "mean_spearman"
            ),
            "027 pooled arm (wave 7), three seeds": arm(
                "wave7_aa_pooled_e200", "mean_spearman"
            ),
            "bilinear ridge, validation-selected embedding (job 1911)": ladder(
                "amino_acids", "B2_bilinear"
            ),
        },
    }


#: (label, head, arm stem, control stem); a stem takes ``_s<seed>.json``.
PAIRS = [
    (
        "betaxanthin: whole-vector store affine against the control",
        "betaxanthin",
        "w134_bxstore_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: two summary columns against the control",
        "betaxanthin",
        "w136_bxratio_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: whole vector, affine decayed and slowed, against the control",
        "betaxanthin",
        "w136_bxstorewd_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: the growth block alone against the whole vector",
        "betaxanthin",
        "w142_bxgrowth_h12l1p1_e200",
        "w134_bxstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: the production block alone against the whole vector",
        "betaxanthin",
        "w142_bxprod_h12l1p1_e200",
        "w134_bxstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: the growth block alone against the control",
        "betaxanthin",
        "w142_bxgrowth_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: the production block alone against the control",
        "betaxanthin",
        "w142_bxprod_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only cell graphs against the control",
        "betaxanthin",
        "w137_bxdel_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: six rule-count columns against the control",
        "betaxanthin",
        "w137_bxgpr_h12l1p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: width 90, two blocks, against the width-12 control",
        "betaxanthin",
        "w139_bxwide_h90l2p1_e200",
        "w134_bxctx_h12l1p1_e200",
    ),
    (
        "betaxanthin: width 90 with the store affine against the width-90 control",
        "betaxanthin",
        "w140_bxwidestore_h90l2p1_e200",
        "w139_bxwide_h90l2p1_e200",
    ),
    (
        "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
        "betaxanthin",
        "w145_bxwidedelstore_h90l2p1_e200",
        "w139_bxwide_h90l2p1_e200",
    ),
    (
        "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-12 one",
        "betaxanthin",
        "w145_bxwidedelstore_h90l2p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
        "betaxanthin",
        "w149_bxwidedel_h90l2p1_e200",
        "w139_bxwide_h90l2p1_e200",
    ),
    (
        "betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs",
        "betaxanthin",
        "w145_bxwidedelstore_h90l2p1_e200",
        "w149_bxwidedel_h90l2p1_e200",
    ),
    (
        "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
        "betaxanthin",
        "w147_bxdelstore40_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
        "betaxanthin",
        "w147_bxdelstore40_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
        "betaxanthin",
        "w148_bxdelstorer15_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store, affine from escape, against deletion-only graphs",
        "betaxanthin",
        "w148_bxdelstorer15_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
        "betaxanthin",
        "w140_bxdelstore_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only graphs with the store affine against the store affine",
        "betaxanthin",
        "w140_bxdelstore_h12l1p1_e200",
        "w134_bxstore_h12l1p1_e200",
    ),
    (
        "amino acids: whole-vector store affine against the control",
        "mulleder19",
        "w135_aastore_h12l1p1_e200",
        "w135_aactx_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only cell graphs against the control",
        "mulleder19",
        "w141_aadel_h12l1p1_e200",
        "w135_aactx_h12l1p1_e200",
    ),
    (
        "amino acids: 19 synthesis ratios and the growth ratio against the control",
        "mulleder19",
        "w138_aaprodstat_h12l1p1_e200",
        "w135_aactx_h12l1p1_e200",
    ),
    (
        "amino acids: growth vector with all 42 summaries against the control",
        "mulleder19",
        "w138_aaprodfull_h12l1p1_e200",
        "w135_aactx_h12l1p1_e200",
    ),
    (
        "amino acids: width 90, two blocks, against the width-12 control",
        "mulleder19",
        "w139_aawide_h90l2p1_e200",
        "w135_aactx_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only graphs with the store affine against deletion-only graphs",
        "mulleder19",
        "w143_aadelstore_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only graphs with the store affine against the store affine",
        "mulleder19",
        "w143_aadelstore_h12l1p1_e200",
        "w135_aastore_h12l1p1_e200",
    ),
    # The essentiality finalists screened on each target (2026-09-23).
    (
        "amino acids: vector with rule counts on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w150_aadelgpr_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: vector with rule counts on deletion-only graphs against the store affine there",
        "mulleder19",
        "w150_aadelgpr_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "amino acids: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w151_aadelhead256_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: solve head (hidden 256) on deletion-only graphs against the store affine there",
        "mulleder19",
        "w151_aadelhead256_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w152_bxdelgpr_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: vector with rule counts on deletion-only graphs against the store affine there",
        "betaxanthin",
        "w152_bxdelgpr_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w153_bxdelhead256_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there",
        "betaxanthin",
        "w153_bxdelhead256_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w154_bxdelmoma_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there",
        "betaxanthin",
        "w154_bxdelmoma_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "amino acids: linear MOMA store on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w155_aadelmoma_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: linear MOMA store on deletion-only graphs against the parsimonious store there",
        "mulleder19",
        "w155_aadelmoma_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w156_bxdelpfba_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there",
        "betaxanthin",
        "w156_bxdelpfba_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w157_bxdelmdf_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there",
        "betaxanthin",
        "w157_bxdelmdf_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "amino acids: driving-force concentration store on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w158_aadelmdf_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: driving-force concentration store on deletion-only graphs against the parsimonious store there",
        "mulleder19",
        "w158_aadelmdf_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w159_bxdelsampled_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there",
        "betaxanthin",
        "w159_bxdelsampled_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store at width 6 against width 12",
        "betaxanthin",
        "w161_bxdelstoreh6_h6l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
        "betaxanthin",
        "w162_bxdelstoredrop_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only store at width 6 against width 12",
        "mulleder19",
        "w163_aadelstoreh6_h6l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w167_bxdelgo_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there",
        "betaxanthin",
        "w167_bxdelgo_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
        "betaxanthin",
        "w168_bxdelgopfba_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO appended against GO alone",
        "betaxanthin",
        "w168_bxdelgopfba_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
        "betaxanthin",
        "w169_bxdelpfbarel_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule",
        "betaxanthin",
        "w169_bxdelpfbarel_h12l1p1_e200",
        "w148_bxdelstorer15_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
        "betaxanthin",
        "w171_bxdelpfbareldrop_h12l1p1_e200",
        "w169_bxdelpfbarel_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
        "betaxanthin",
        "w171_bxdelpfbareldrop_h12l1p1_e200",
        "w162_bxdelstoredrop_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
        "betaxanthin",
        "w172_bxdelstoredrop5_h12l1p1_e200",
        "w162_bxdelstoredrop_h12l1p1_e200",
    ),
    (
        "betaxanthin: deletion-only store at dropout 0.5 against 0.1",
        "betaxanthin",
        "w172_bxdelstoredrop5_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
        "betaxanthin",
        "w166_bxdelmomarel_h12l1p1_e200",
        "w154_bxdelmoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
        "betaxanthin",
        "w166_bxdelmomarel_h12l1p1_e200",
        "w148_bxdelstorer15_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
        "betaxanthin",
        "w164_bxdelmdfpoly_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there",
        "betaxanthin",
        "w164_bxdelmdfpoly_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point",
        "betaxanthin",
        "w164_bxdelmdfpoly_h12l1p1_e200",
        "w157_bxdelmdf_h12l1p1_e200",
    ),
    (
        "amino acids: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w165_aadelmdfpoly_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: driving-force polytope mean on deletion-only graphs against the parsimonious store there",
        "mulleder19",
        "w165_aadelmdfpoly_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "amino acids: driving-force polytope mean on deletion-only graphs against the driving-force point",
        "mulleder19",
        "w165_aadelmdfpoly_h12l1p1_e200",
        "w158_aadelmdf_h12l1p1_e200",
    ),
    (
        "amino acids: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w170_aadelgo_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: GO multi-hot in the affine slot on deletion-only graphs against the parsimonious store there",
        "mulleder19",
        "w170_aadelgo_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force polytope mean with the spread against the mean alone",
        "betaxanthin",
        "w174_bxdelmdfpolyspread_h12l1p1_e200",
        "w164_bxdelmdfpoly_h12l1p1_e200",
    ),
    (
        "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
        "betaxanthin",
        "w174_bxdelmdfpolyspread_h12l1p1_e200",
        "w140_bxdelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO affine released on escape against the GO affine",
        "betaxanthin",
        "w175_bxdelgorel_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO affine released on escape against the GLPK store under the rule",
        "betaxanthin",
        "w175_bxdelgorel_h12l1p1_e200",
        "w148_bxdelstorer15_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO affine at dropout 0.3 against the GO affine",
        "betaxanthin",
        "w176_bxdelgodrop_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
        "betaxanthin",
        "w176_bxdelgodrop_h12l1p1_e200",
        "w162_bxdelstoredrop_h12l1p1_e200",
    ),
    (
        "amino acids: HiGHS store with GO appended against GO alone",
        "mulleder19",
        "w177_aadelgopfba_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: HiGHS store with GO appended against the parsimonious store alone",
        "mulleder19",
        "w177_aadelgopfba_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
        "betaxanthin",
        "w178_bxdelgopfbadrop_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
        "betaxanthin",
        "w178_bxdelgopfbadrop_h12l1p1_e200",
        "w176_bxdelgodrop_h12l1p1_e200",
    ),
    (
        "amino acids: HiGHS store with GO appended at dropout 0.3 against 0.1",
        "mulleder19",
        "w179_aadelgopfbadrop_h12l1p1_e200",
        "w177_aadelgopfba_h12l1p1_e200",
    ),
    (
        "amino acids: HiGHS store with GO appended at dropout 0.3 against the store at 0.3",
        "mulleder19",
        "w179_aadelgopfbadrop_h12l1p1_e200",
        "w173_aadelstoredrop_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
        "betaxanthin",
        "w180_bxdelgopfbapoly_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
        "betaxanthin",
        "w180_bxdelgopfbapoly_h12l1p1_e200",
        "w164_bxdelmdfpoly_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
        "betaxanthin",
        "w183_bxdelgopfbapolyspread_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
        "betaxanthin",
        "w183_bxdelgopfbapolyspread_h12l1p1_e200",
        "w180_bxdelgopfbapoly_h12l1p1_e200",
    ),
    (
        "betaxanthin: MOMA store with GO appended against GO alone",
        "betaxanthin",
        "w184_bxdelgomoma_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
        "betaxanthin",
        "w184_bxdelgomoma_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
        "betaxanthin",
        "w188_bxdelgopfbawd_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
        "betaxanthin",
        "w189_bxdelgopfbawd1_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS store with GO at affine decay 1.0 against decay 0.1",
        "betaxanthin",
        "w189_bxdelgopfbawd1_h12l1p1_e200",
        "w188_bxdelgopfbawd_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
        "betaxanthin",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
        "betaxanthin",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
        "w184_bxdelgomoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO multi-hot in the affine slot against the control, twelve seeds",
        "betaxanthin",
        "w167_bxdelgo_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "amino acids: HiGHS store with GO appended against GO alone, twelve seeds",
        "mulleder19",
        "w177_aadelgopfba_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: HiGHS store with GO appended against GO alone, twelve seeds",
        "betaxanthin",
        "w168_bxdelgopfba_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO, twelve seeds",
        "betaxanthin",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
        "betaxanthin",
        "w197_bxdelgopfbamomaaapool_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families against the pool with GO and the HiGHS vector",
        "betaxanthin",
        "w197_bxdelgopfbamomaaapool_h12l1p1_e200",
        "w195_bxdelgopfbaaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
        "betaxanthin",
        "w200_bxdell2_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: minimum-norm flux store against the control",
        "betaxanthin",
        "w200_bxdell2_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
        "betaxanthin",
        "w201_bxdelgol2_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: minimum-norm flux store with GO against GO alone",
        "betaxanthin",
        "w201_bxdelgol2_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool alone in the affine against the control",
        "betaxanthin",
        "w198_bxdelaapool_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
        "betaxanthin",
        "w198_bxdelaapool_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
        "betaxanthin",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
        "w198_bxdelaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
        "betaxanthin",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool with GO against GO alone",
        "betaxanthin",
        "w194_bxdelgoaapool_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
        "betaxanthin",
        "w195_bxdelgopfbaaapool_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against the pool with GO",
        "betaxanthin",
        "w195_bxdelgopfbaaapool_h12l1p1_e200",
        "w194_bxdelgoaapool_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score with GO against GO alone",
        "mulleder19",
        "w202_aadelgobxscore_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score with GO against the control",
        "mulleder19",
        "w202_aadelgobxscore_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
        "betaxanthin",
        "w204_bxdelgoaapoolmlp_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
        "betaxanthin",
        "w204_bxdelgoaapoolmlp_h12l1p1_e200",
        "w194_bxdelgoaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
        "betaxanthin",
        "w204_bxdelgoaapoolmlp_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control, twelve seeds",
        "betaxanthin",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
        "betaxanthin",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
        "betaxanthin",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the pool in the linear affine",
        "betaxanthin",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        "w198_bxdelaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
        "betaxanthin",
        "w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200",
        "w197_bxdelgopfbamomaaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families through a hidden layer of 64 against GO with the pool through the same layer",
        "betaxanthin",
        "w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200",
        "w204_bxdelgoaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families through a hidden layer of 64 against the pool alone through the same layer",
        "betaxanthin",
        "w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
        "betaxanthin",
        "w209_bxdelaapoolmlp_h12l1p1_e400",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
        "betaxanthin",
        "w210_bxdelaapoolpfbamlp_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
        "betaxanthin",
        "w210_bxdelaapoolpfbamlp_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
        "betaxanthin",
        "w211_bxdelaapoolmlp512_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
        "betaxanthin",
        "w211_bxdelaapoolmlp512_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
        "betaxanthin",
        "w213_bxdelaapoolmlp128_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 256",
        "betaxanthin",
        "w213_bxdelaapoolmlp128_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64, twelve seeds",
        "betaxanthin",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
        "betaxanthin",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the control, twelve seeds",
        "betaxanthin",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 with a budget of 400 epochs against the budget of 200",
        "betaxanthin",
        "w215_bxdelaapoolmlp256_h12l1p1_e400",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns through one shared layer of 64",
        "betaxanthin",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
        "w204_bxdelgoaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 64, against the pool alone through 64",
        "betaxanthin",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
        "w199_bxdelaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns in the linear affine",
        "betaxanthin",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
        "w194_bxdelgoaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
        "betaxanthin",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 256, against the pool alone through 256",
        "betaxanthin",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 256, against the two slots at 64",
        "betaxanthin",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 256, against GO alone",
        "betaxanthin",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w197_bxdelgopfbamomaaapool_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families with two slots against the same columns through one shared layer of 64",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families with two slots against GO linear beside the pool through 256",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families with two slots against the pool alone through 256",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
        "betaxanthin",
        "w219_bxdelgolinaapoolmlp128_h12l1p1_e200",
        "w213_bxdelaapoolmlp128_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
        "betaxanthin",
        "w219_bxdelgolinaapoolmlp128_h12l1p1_e200",
        "w216_bxdelgolinaapoolmlp_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
        "betaxanthin",
        "w219_bxdelgolinaapoolmlp128_h12l1p1_e200",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
        "betaxanthin",
        "w220_bxdelaapoolmlp256drop_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
        "betaxanthin",
        "w221_bxdelaapoolmlp256std_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "amino acids: the betaxanthin score through a hidden layer of 256 against the score in the linear affine",
        "mulleder19",
        "w225_aadelbxscoremlp256_h12l1p1_e200",
        "w203_aadelbxscore_h12l1p1_e200",
    ),
    (
        "amino acids: the betaxanthin score through a hidden layer of 256 against GO alone",
        "mulleder19",
        "w225_aadelbxscoremlp256_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: Cooper's peaks through a hidden layer of 256 against the same columns linear",
        "mulleder19",
        "w226_aadelcooperpoolmlp256_h12l1p1_e200",
        "w227_aadelcooperpool_h12l1p1_e200",
    ),
    (
        "amino acids: Cooper's peaks through a hidden layer of 256 against GO alone",
        "mulleder19",
        "w226_aadelcooperpoolmlp256_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: Cooper's peaks in the linear affine against GO alone",
        "mulleder19",
        "w227_aadelcooperpool_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: Cooper's peaks in the linear affine against the control",
        "mulleder19",
        "w227_aadelcooperpool_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
        "betaxanthin",
        "w229_bxdelaapoolmlp256rel_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
        "betaxanthin",
        "w230_bxdelcooperpoolmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
        "betaxanthin",
        "w230_bxdelcooperpoolmlp256_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
        "betaxanthin",
        "w231_bxdelaapoolmlp256w36_h36l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36",
        "betaxanthin",
        "w232_bxdelaapoolmlp256ss_h36l2p1_e200",
        "w231_bxdelaapoolmlp256w36_h36l1p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks",
        "betaxanthin",
        "w233_bxdelaapoolmlp256skl_h36l2p1_e200",
        "w232_bxdelaapoolmlp256ss_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks",
        "betaxanthin",
        "w234_bxdelaapoolmlp256smask_h36l2p1_e200",
        "w232_bxdelaapoolmlp256ss_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked",
        "betaxanthin",
        "w235_bxdelaapoolmlp256smm_h36l3p1_e200",
        "w234_bxdelaapoolmlp256smask_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read",
        "betaxanthin",
        "w235_bxdelaapoolmlp256smm_h36l3p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight",
        "betaxanthin",
        "w236_bxdelaapoolmlp256skl10_h36l2p1_e200",
        "w233_bxdelaapoolmlp256skl_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks",
        "betaxanthin",
        "w236_bxdelaapoolmlp256skl10_h36l2p1_e200",
        "w232_bxdelaapoolmlp256ss_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight",
        "betaxanthin",
        "w237_bxdelaapoolmlp256skl100_h36l2p1_e200",
        "w233_bxdelaapoolmlp256skl_h36l2p1_e200",
    ),
    (
        "betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks",
        "betaxanthin",
        "w237_bxdelaapoolmlp256skl100_h36l2p1_e200",
        "w232_bxdelaapoolmlp256ss_h36l2p1_e200",
    ),
    (
        "betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256",
        "betaxanthin",
        "w238_bxdelgraphlabelmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: the graphs' one-hop label-propagation store through 256 against the control",
        "betaxanthin",
        "w238_bxdelgraphlabelmlp256_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone",
        "betaxanthin",
        "w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: the pool with the label-propagation store appended, through 256, against the label-propagation store alone",
        "betaxanthin",
        "w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200",
        "w238_bxdelgraphlabelmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone",
        "betaxanthin",
        "w240_bxdelpfba4m_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init",
        "betaxanthin",
        "w241_bxdelpfbadef_h12l1p1_e200",
        "w156_bxdelpfba_h12l1p1_e200",
    ),
    (
        "betaxanthin: the HiGHS parsimonious store under default affine initialization against the four-media control",
        "betaxanthin",
        "w241_bxdelpfbadef_h12l1p1_e200",
        "w240_bxdelpfba4m_h12l1p1_e200",
    ),
    (
        "betaxanthin: the pool with the label-propagation store appended, through 256, against one block of width 36",
        "betaxanthin",
        "w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200",
        "w231_bxdelaapoolmlp256w36_h36l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
        "betaxanthin",
        "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
        "betaxanthin",
        "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
        "betaxanthin",
        "w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200",
        "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against a tenth",
        "betaxanthin",
        "w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200",
        "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200",
    ),
    (
        "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against the pool alone through 256",
        "betaxanthin",
        "w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: three families with two slots against GO alone, twelve seeds",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: three families with two slots against the HiGHS store with GO, twelve seeds",
        "betaxanthin",
        "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
        "w168_bxdelgopfba_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "amino acids: measured betaxanthin score alone in the affine against GO alone, twelve seeds",
        "mulleder19",
        "w203_aadelbxscore_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "amino acids: measured betaxanthin score alone in the affine against the control, twelve seeds",
        "mulleder19",
        "w203_aadelbxscore_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "amino acids: GO alone standardized against GO alone under log1p",
        "mulleder19",
        "w208_aadelgostd_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score with GO against GO alone under the same standardized encoding",
        "mulleder19",
        "w202_aadelgobxscore_h12l1p1_e200",
        "w208_aadelgostd_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score alone in the affine against the score with GO",
        "mulleder19",
        "w203_aadelbxscore_h12l1p1_e200",
        "w202_aadelgobxscore_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score alone in the affine against GO alone",
        "mulleder19",
        "w203_aadelbxscore_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: measured betaxanthin score alone in the affine against the control",
        "mulleder19",
        "w203_aadelbxscore_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: GO multi-hot in the affine slot against the control, twelve seeds",
        "mulleder19",
        "w170_aadelgo_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "amino acids: GO affine at dropout 0.3 against 0.1",
        "mulleder19",
        "w182_aadelgodrop_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "amino acids: GO affine at dropout 0.3 against the store at 0.3",
        "mulleder19",
        "w182_aadelgodrop_h12l1p1_e200",
        "w173_aadelstoredrop_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only store at dropout 0.3 against 0.1",
        "mulleder19",
        "w173_aadelstoredrop_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only store at dropout 0.3 against deletion-only graphs",
        "mulleder19",
        "w173_aadelstoredrop_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: deletion-only graphs for 300 epochs against 200",
        "mulleder19",
        "w146_aadel_h12l1p1_e300",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
        "mulleder19",
        "w160_aadelsampled_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: sampled-cone mean store on deletion-only graphs against the parsimonious store there",
        "mulleder19",
        "w160_aadelsampled_h12l1p1_e200",
        "w143_aadelstore_h12l1p1_e200",
    ),
    # The inductive program (2026-09-28): co-heads and feature propagation.
    (
        "betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds",
        "betaxanthin",
        "w242_bxdelgopfbamomaaux_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
        tuple(range(2700, 2712)),
    ),
    (
        "betaxanthin: GO alone with the Mulleder co-head against without",
        "betaxanthin",
        "w243_bxdelgoaux_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: the genotype-only control with the Mulleder co-head against without",
        "betaxanthin",
        "w244_bxdelaux_h12l1p1_e200",
        "w137_bxdel_h12l1p1_e200",
    ),
    (
        "betaxanthin: the GO neighbor-propagation store (inductive) against GO alone",
        "betaxanthin",
        "w245_bxdelgonbr_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO with its neighbor propagation appended against GO alone",
        "betaxanthin",
        "w246_bxdelgogonbr_h12l1p1_e200",
        "w167_bxdelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: GO with its neighbor propagation appended against the propagation alone",
        "betaxanthin",
        "w246_bxdelgogonbr_h12l1p1_e200",
        "w245_bxdelgonbr_h12l1p1_e200",
    ),
    (
        "betaxanthin: the GO neighbor-propagation store through 256 against the linear affine",
        "betaxanthin",
        "w247_bxdelgonbrmlp256_h12l1p1_e200",
        "w245_bxdelgonbr_h12l1p1_e200",
    ),
    (
        "betaxanthin: the GO neighbor-propagation store through 256 against the pool through 256",
        "betaxanthin",
        "w247_bxdelgonbrmlp256_h12l1p1_e200",
        "w206_bxdelaapoolmlp256_h12l1p1_e200",
    ),
    (
        "amino acids: the control with the betaxanthin co-head (inductive) against without",
        "mulleder19",
        "w232_aadelauxbx_h12l1p1_e200",
        "w141_aadel_h12l1p1_e200",
    ),
    (
        "amino acids: the control with the betaxanthin co-head against the betaxanthin score as an input",
        "mulleder19",
        "w232_aadelauxbx_h12l1p1_e200",
        "w225_aadelbxscoremlp256_h12l1p1_e200",
    ),
    (
        "amino acids: GO with its neighbor propagation appended (inductive) against GO alone",
        "mulleder19",
        "w233_aadelgogonbr_h12l1p1_e200",
        "w170_aadelgo_h12l1p1_e200",
    ),
    (
        "betaxanthin: the perturbation operator with two ReZero blocks against the inductive incumbent",
        "betaxanthin",
        "w248_bxdelgopfbamomapert2_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the perturbation operator with the Hadamard pair term against the inductive incumbent",
        "betaxanthin",
        "w249_bxdelgopfbamomahad_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the perturbation operator with the null sink against the inductive incumbent",
        "betaxanthin",
        "w250_bxdelgopfbamomasink_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the perturbation operator with depth, the pair term and the null sink against the inductive incumbent",
        "betaxanthin",
        "w251_bxdelgopfbamomapertall_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the species language-model composite embedding against ProtT5 alone on the inductive incumbent",
        "betaxanthin",
        "w252_bxdelgopfbamomacomp_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the composite with ESM2 added against ProtT5 alone on the inductive incumbent",
        "betaxanthin",
        "w253_bxdelgopfbamomacompesm_h12l1p1_e200",
        "w190_bxdelgopfbamoma_h12l1p1_e200",
    ),
    (
        "betaxanthin: the composite with ESM2 added against the composite",
        "betaxanthin",
        "w253_bxdelgopfbamomacompesm_h12l1p1_e200",
        "w252_bxdelgopfbamomacomp_h12l1p1_e200",
    ),
]


class Cell(BaseModel):
    """One cell's held-out read at its selected epoch."""

    seed: int
    select_epoch: int
    select_val_pearson: float
    spearman: float
    pearson: float
    top50_enrichment: float | None = None
    top50_hits: float | None = None


class Arm(BaseModel):
    """One arm over its seeds."""

    stem: str
    cells: list[Cell]
    mean_spearman: float
    sd_spearman: float
    mean_top50_enrichment: float | None = None


class Pair(BaseModel):
    """An arm against its control, paired by seed."""

    name: str
    head: str
    arm: Arm
    control: Arm
    n: int
    mean_difference: float
    sd_difference: float
    n_positive: int
    differences: list[float]


class Report(BaseModel):
    """The readout."""

    comparators: dict[str, dict[str, float]]
    pairs: list[Pair]
    note: str


def read_cell(stem: str, seed: int, head: str) -> Cell | None:
    path = osp.join(RESULTS_BY_HEAD[head], f"{stem}_s{seed}.json")
    if not osp.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    t = d["test"]["pinned"]
    if head == "mulleder19":
        return Cell(
            seed=seed,
            select_epoch=int(d["select_epoch"]),
            select_val_pearson=float(d["select_val_pearson"]),
            spearman=float(t["mean_spearman"]),
            pearson=float(t["mean_pearson"]),
        )
    return Cell(
        seed=seed,
        select_epoch=int(d["select_epoch"]),
        select_val_pearson=float(d["select_val_pearson"]),
        spearman=float(t["spearman"]),
        pearson=float(t["pearson"]),
        top50_enrichment=float(t["top50"]["enrichment"]),
        top50_hits=float(t["top50"]["hits"]),
    )


def read_arm(stem: str, head: str, seeds: tuple[int, ...]) -> Arm:
    cells = [c for s in seeds if (c := read_cell(stem, s, head)) is not None]
    sp = np.array([c.spearman for c in cells]) if cells else np.array([np.nan])
    enr = [c.top50_enrichment for c in cells if c.top50_enrichment is not None]
    return Arm(
        stem=stem,
        cells=cells,
        mean_spearman=float(sp.mean()),
        sd_spearman=float(sp.std(ddof=1)) if len(sp) > 1 else float("nan"),
        mean_top50_enrichment=float(np.mean(enr)) if enr else None,
    )


def main() -> None:
    pairs: list[Pair] = []
    for entry in PAIRS:
        name, head, arm_stem, ctrl_stem = entry[:4]
        # A pair may carry its own seed set as a fifth element; the extension rounds
        # (127, 164, 168, 172) read an arm and its control over twelve seeds.
        seeds = entry[4] if len(entry) > 4 else SEEDS
        arm = read_arm(arm_stem, head, seeds)
        ctrl = read_arm(ctrl_stem, head, seeds)
        by_seed = {c.seed: c for c in ctrl.cells}
        diffs = [
            c.spearman - by_seed[c.seed].spearman
            for c in arm.cells
            if c.seed in by_seed
        ]
        if not diffs:
            print(f"{name}: no paired seeds yet")
            continue
        d = np.array(diffs)
        pair = Pair(
            name=name,
            head=head,
            arm=arm,
            control=ctrl,
            n=len(d),
            mean_difference=float(d.mean()),
            sd_difference=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            n_positive=int((d > 0).sum()),
            differences=[float(x) for x in d],
        )
        pairs.append(pair)
        print(
            f"{name}: n {pair.n} arm {arm.mean_spearman:.4f} (sd {arm.sd_spearman:.4f}) "
            f"control {ctrl.mean_spearman:.4f} (sd {ctrl.sd_spearman:.4f}) diff "
            f"{pair.mean_difference:+.4f} sd {pair.sd_difference:.4f} positive {pair.n_positive}/{pair.n}"
            + (
                f"; top-50 enrichment {arm.mean_top50_enrichment:.2f} vs {ctrl.mean_top50_enrichment:.2f}"
                if arm.mean_top50_enrichment is not None
                and ctrl.mean_top50_enrichment is not None
                else ""
            )
        )
        for c in arm.cells:
            k = by_seed.get(c.seed)
            print(
                f"   s{c.seed}: arm {c.spearman:+.4f} (ep {c.select_epoch}, val r {c.select_val_pearson:+.3f})"
                + (
                    f" control {k.spearman:+.4f} (ep {k.select_epoch}, val r {k.select_val_pearson:+.3f})"
                    if k
                    else ""
                )
            )
    comps = comparators()
    # One report per study, in that study's results tree.
    for head, results in RESULTS_BY_HEAD.items():
        report = Report(
            comparators={head: comps[head]},
            pairs=[p for p in pairs if p.head == head],
            note=(
                "Pinned-test Spearman at the validation-selected epoch (centered five-epoch "
                "moving average of validation Pearson), paired by seed; the amino-acid "
                "statistic is the mean per-column Spearman over the 19 amino acids. "
                "Comparators are read from 027's result files on the same split and "
                "statistic."
            ),
        )
        out = osp.join(results, "transfer_readout.json")
        with open(out, "w") as fh:
            fh.write(report.model_dump_json(indent=2))
        print(out)


if __name__ == "__main__":
    main()
