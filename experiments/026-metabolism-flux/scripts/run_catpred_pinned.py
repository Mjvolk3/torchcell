# experiments/026-metabolism-flux/scripts/run_catpred_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_catpred_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_catpred_pinned.py

r"""Run CatPred inside its pinned prefix and write predictions plus their variance.

WHAT CATPRED IS
---------------
Boorla and Maranas (2025), *Nature Communications* 16:2072, doi
10.1038/s41467-025-57215-9, citation key ``boorlaCatPredComprehensiveFramework2025``. A
fork of chemprop: the substrate is a directed message-passing network over its 2D
atom-bond graph, the enzyme is a learned residue embedding carrying rotary positional
information through multi-head self-attention, concatenated with ESM-2 650M features and
reduced by an attentive pooling layer. The two representations meet in a fully connected
head. No EC number is an input, and the released structure (E-GNN) branch is used only by
the :math:`K_i` model, which is not mirrored here.

THE REASON IT IS WORTH ADDING TO A SET OF SEVEN
------------------------------------------------
It is the only predictor in the mirror that emits a **per-prediction variance**. The head
outputs a mean and a variance and is trained under a negative log-likelihood loss, and ten
independently initialized copies form an ensemble, so one forward pass over the ensemble
gives three numbers rather than one:

* the prediction, the arithmetic mean of the ten means, in :math:`\log_{10}`;
* the **aleatoric** variance, the mean of the ten predicted variances, which is the noise
  the training labels themselves carry;
* the **epistemic** variance, the variance of the ten means, which is how much the
  ensemble disagrees and therefore how far the input sits from the training distribution.

``predict.py`` writes the total variance as ``<target>_mve_uncal_var`` and the ten member
predictions as ``<target>_model_i``. This script recomputes all three terms from those
columns rather than reading the ``SD_*`` columns the repository's own postprocessing adds,
because that postprocessing broadcasts row 0's uncertainty to every row: in the shipped
``demo/batch_kcat_input_output.csv`` rows 0 and 1 carry different predictions and
identical ``SD_total``, ``SD_aleatoric`` and ``SD_epistemic``.

THE THREE CONVENTIONS THAT DECIDE WHETHER THE NUMBERS MEAN ANYTHING
--------------------------------------------------------------------
* **The output is** :math:`\log_{10}`. The targets are named ``log10kcat_max`` and
  ``log10km_mean`` in the checkpoints' own ``args.json``, and ``--selftest`` reproduces
  the repository's shipped demo output to five decimals, which is what settles it.
* **Units are the training tables'**: :math:`k_{cat}` in 1/s and :math:`K_M` in mM, the
  BRENDA conventions and already the canonical units of ``KineticPrediction``.
* **Sequences are truncated to 2,048 residues** by ``catpred/data/esm_utils.py``. That is
  the model's cap rather than a choice made here, and it affects 6 of the 1,153 GEM
  proteins.

WHAT THIS RUN IS NOT
--------------------
CatPred's :math:`k_{cat}` model was trained on the **concatenated** SMILES of every
reactant, not on a single substrate. The pinned pair table gives one substrate per row,
which is what the other seven predictors were shown, so that is what CatPred is shown too:
the comparison is like-for-like and the deviation from the model's training convention is
real and is recorded here rather than hidden. :math:`K_M` has no such gap, since the
authors use the single relevant substrate for it.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/run_catpred_pinned.py --selftest
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import os
import os.path as osp
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/catpred"
#: CatPred needs ``typed-argument-parser``, ``descriptastorus``, ``pandas-flavor``,
#: ``tensorboardX``, ``ipdb`` and ``rotary_embedding_torch==0.6.5``, none of which
#: torchcell depends on. They live in their own prefix rather than in the project
#: environment, the same arrangement EITLEM uses for ``fair-esm``. The rotary pin is not
#: cosmetic: a later release changes the cached-frequency shape and the released
#: checkpoints fail to run at all.
PINNED_SITE_PACKAGES = "/scratch/projects/torchcell-scratch/envs/catpred/site-packages"

#: The released production ensembles. ``fold_0`` holds ``model_0`` to ``model_9``, the ten
#: differently seeded copies whose spread is the epistemic term.
CHECKPOINTS = {"k_cat": "pretrained/production/kcat", "K_M": "pretrained/production/km"}
#: The target column each checkpoint predicts, from its own ``args.json``.
TARGETS = {"k_cat": "log10kcat_max", "K_M": "log10km_mean"}
ENSEMBLE_SIZE = 10
ESM_MAX_RESIDUES = 2048
VALID_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: The first row of the repository's own ``demo/batch_kcat.csv``, and the value its
#: shipped ``demo/batch_kcat_input_output.csv`` records for it. Reproducing this is the
#: check that the checkpoint, the ESM-2 encoder and the output transform are the ones the
#: authors released; a wrong log base would miss it by a power.
SELFTEST_SMILES = "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O.O"
SELFTEST_SEQUENCE = (
    "MTQQRQVASGSPHTVLAQQLISPLLLNHTNSKNPVTVVTIDGHSLSIEQLIRSTLGVGLGQKTLSAQPASTTTKNAVEFQL"
    "TNDRETIDRINESLQFMRARTGESTYGVTTGFGAAATTRTRHVQELQVSIVEHLLAGITGVDLSQLANQLKAPIHPSQPLT"
    "PNVMPESIVRAAILIRVNSLTRGHSAVRLELLHGLLSLLNLSITPVVPLRGTISASGDLAPLAYVAGALCAHPDVYVIDRS"
    "LKEPKVLSSAECLKAHGITPLVLGPKEGLAIANGTAFSAAAASMAMFQAHTLGLLSQVLTAMTVEAMVGQIGAFHPFIHQT"
    "ARPHAGQIEVAKNIYRLLQTSQLLESHDEMADHDLDRERSKQILRQDRYPLRTAPQWIGPQLEDLLVAHQAIAKELNSTTD"
    "NPLVDVENGILHHGGNFQATSVATSMEKTRLAIAAIGKIIFAQVTELNNSFMNKGLPSCLNGHEPSTNYHTKGLDTSCAAY"
    "CAELQYLAGPVTTHVQSAEGHNQSINSLAFISARKTLEAIEILKMLMASHLYCLCQALDLRVLEIRLKNKLSGIFSETVEQ"
    "NFGAQLSKKRMDAVVDKMIDRFWFRREMTASLDTKERVRDGLEGCVAVVMEAYELEKKQMPVGAMASFLSSSADMIAACLE"
    "GLRAEGIDHKTVDFLGGTRPLYEFVRNSLGIRARRGDVVEGTPGPTIGGMVNKIFRSLQFELLPTDQPQQFLQVLLSLFES"
    "PPAIHGDSNSKDSLCFLSPLAELDNFLNTNIFSTPFACAVSVPPVSFFLFFVLLPLALSQSSPSQLRSPSVSSFTLSTPLL"
    "FFLPPQDPPSLHYVTLNLCSSPTSQPIDFQFNNILIVSNSSDNKTSSPDQLPARSKKSIISHQSSSGDYSDLRGGFASVSL"
    "VNSVGLWISIQSPNLTKSWDFQLGVSKETPLHALNGFPAFKLDDTDDTNALMTAADPSLFQPLTDKTPLSLQQFVPMILPT"
    "PDTLQIGLSGSLCYLQSLVAAQSQDFKTHVAQLSITSSLTTRTTGAGSFSSQPTGNATDHDYRISDQAAGLRTQYLINNLR"
    "PALNYSAWLFQPGPLVDNVQTGKLWPYINFRTKISKNCRLVHDLPFCPSVAYAVPSSPSQTTSSLIDMYNQTVSVHLANFR"
    "TVVSTYPCNNDTSGRYSFVTGCGDCLRAYTDWVCAVGMPRCTDAPARPVVAAAATETQVVFTRTEPSASRTPNMTDKSVYP"
    "YAELPPCASLCTLVAATCPPLIGWTCPLPGVSLGSSYRDSRPLSPLELSGGHQSLAGFGGDRAQDRFGNVFCNALESDIVY"
    "ARMGSASPLRISVQTLASLLVAQSFFFFFFLLA"
)
SELFTEST_EXPECTED_LOG10 = 0.1875500416063556
SELFTEST_TOLERANCE = 1e-4


class CatPredRun(BaseModel):
    """Everything that decides what a CatPred value is, recorded before the run."""

    model_config = ConfigDict(extra="forbid")

    inputs_path: str
    output_path: str
    device: str = "cpu"
    mirror: str = MIRROR
    parameters: list[str] = Field(default_factory=lambda: ["k_cat", "K_M"])
    ensemble_size: int = ENSEMBLE_SIZE
    esm_max_residues: int = ESM_MAX_RESIDUES
    #: sha256 of the ensemble member the mirror manifest pins as this run's anchor. One
    #: hash names the build; the manifest resolves the other nineteen checkpoints.
    weights_sha256: str


class CatPredResult(BaseModel):
    """What one run produced, written beside the parquet."""

    model_config = ConfigDict(extra="forbid")

    run: CatPredRun
    built_at: str
    runtime_seconds: float
    n_input_rows: int
    n_distinct_jobs: int
    n_predicted: dict[str, int]
    n_failed: dict[str, int]
    failure_reasons: dict[str, int]
    value_median: dict[str, float]
    sd_total_median_decades: dict[str, float]
    sd_aleatoric_median_decades: dict[str, float]
    sd_epistemic_median_decades: dict[str, float]
    n_sequences_truncated: int


def sha256_of(path: str) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def anchor_weights_sha256() -> str:
    """The pinned sha256 of ``k_cat`` ensemble member 0, this run's provenance anchor."""
    with open(osp.join(MIRROR, "manifest.json")) as handle:
        manifest = json.load(handle)
    wanted = f"{CHECKPOINTS['k_cat']}/fold_0/model_0/model.pt"
    return str(next(f["sha256"] for f in manifest["files"] if f["rel_path"] == wanted))


def _disable_rdkit_logs() -> None:
    """Silence rdkit's C++ logger before a parse.

    The same typed accessor ``kinetics_pipeline._disable_rdkit_logs`` uses, repeated here
    rather than imported: this runner is a subprocess entry point and deliberately imports
    nothing from torchcell, so that its dependency set is CatPred's and not the project's.
    ``rdkit-stubs`` does not declare ``DisableLog``; the name is re-exported from the
    compiled ``rdBase`` module at runtime.
    """
    from rdkit import RDLogger

    disable = cast(Callable[[str], None], getattr(RDLogger, "DisableLog"))  # noqa: B009
    disable("rdApp.*")


def canonical_smiles(smiles: str, parameter: str) -> str | None:
    """RDKit canonical SMILES, with CatPred's own multi-component ordering for k_cat.

    ``catpred/inference/service.py`` canonicalizes and then, for ``kcat`` only, sorts the
    dot-separated fragments so that a multi-component input has one spelling. Applying the
    same rule here is what makes a value here equal a value from the authors' own CLI.
    """
    from rdkit import Chem

    _disable_rdkit_logs()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    canonical = Chem.MolToSmiles(molecule)
    if parameter == "k_cat" and "." in canonical:
        canonical = ".".join(sorted(canonical.split(".")))
    return canonical


def sequence_key(sequence: str) -> str:
    """Content address for a protein, used as the unique ``pdbpath`` CatPred requires.

    CatPred keys its protein records on the BASENAME of ``pdbpath`` and refuses a basename
    reused for two sequences. No structure file is read for k_cat or K_M, so the honest
    value is a hash of the sequence: unique by construction, and it never suggests a PDB
    that does not exist.
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def _patch_torch_load() -> None:
    """Restore full unpickling, which the released checkpoints require.

    Every ``model.pt`` here carries an ``argparse.Namespace`` of the training arguments,
    and torch 2.6 made ``weights_only=True`` the default, so ``load_args`` raises before
    a single weight is read. These are pinned bytes from the authors' own release, hashed
    in the mirror manifest, so restoring them fully is the correct read.
    """
    torch.load = functools.partial(torch.load, weights_only=False)


def _run_predict(
    jobs: pd.DataFrame, parameter: str, device: str, work: str
) -> pd.DataFrame:
    """Invoke CatPred's own prediction entry point over one parameter's job table.

    ``jobs`` carries the columns CatPred requires (``SMILES``, ``sequence``, ``pdbpath``)
    and nothing else, one row per distinct (sequence, canonical SMILES) pair.
    """
    if MIRROR not in sys.path:
        sys.path.insert(0, MIRROR)
    # ``catpred/data/esm_utils.py`` reads this at import time, so it is set BEFORE the
    # first import and only when there is no GPU to place the encoder on.
    if device == "cpu":
        os.environ["PROTEIN_EMBED_USE_CPU"] = "1"
    _patch_torch_load()
    from catpred.train.make_predictions import catpred_predict

    input_csv = osp.join(work, f"{parameter}_input.csv")
    records_path = osp.join(work, f"{parameter}_records.json.gz")
    output_csv = osp.join(work, f"{parameter}_output.csv")
    jobs.to_csv(input_csv, index=False)
    records = {
        row.pdbpath: {"name": row.pdbpath, "seq": row.sequence}
        for row in jobs.itertuples()
    }
    with gzip.open(records_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)

    argv = [
        "predict.py",
        "--test_path",
        input_csv,
        "--preds_path",
        output_csv,
        "--checkpoint_dir",
        osp.join(MIRROR, CHECKPOINTS[parameter]),
        "--uncertainty_method",
        "mve",
        "--smiles_column",
        "SMILES",
        "--individual_ensemble_predictions",
        "--protein_records_path",
        records_path,
    ]
    if device == "cpu":
        argv.append("--no_cuda")
    else:
        argv += ["--gpu", device.split(":")[-1]]
    saved, sys.argv = sys.argv, argv
    try:
        catpred_predict()
    finally:
        sys.argv = saved
    return pd.read_csv(output_csv)


def n_heavy_atoms(smiles: str) -> int:
    """Heavy-atom count of a canonical SMILES, which is what the D-MPNN needs at least one of."""
    from rdkit import Chem

    _disable_rdkit_logs()
    molecule = Chem.MolFromSmiles(smiles)
    return 0 if molecule is None else int(molecule.GetNumHeavyAtoms())


def _decompose_uncertainty(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    r"""Split CatPred's total variance into its aleatoric and epistemic halves.

    ``<target>_mve_uncal_var`` is the TOTAL predictive variance chemprop reports for an
    ``mve`` ensemble. The epistemic half is the population variance of the ten member
    means, which the ``<target>_model_i`` columns carry, and the aleatoric half is what is
    left. Both are variances of a :math:`\log_{10}` quantity, so their square roots are in
    decades and are directly comparable to the spread statistics the other predictors are
    judged by.
    """
    members = frame[[f"{target}_model_{i}" for i in range(ENSEMBLE_SIZE)]].to_numpy()
    total_var = frame[f"{target}_mve_uncal_var"].to_numpy(dtype=float)
    epistemic_var = members.var(axis=1)
    # Floated at zero rather than allowed negative: the two terms are computed by
    # different code paths in chemprop and round-off can put their difference a few
    # 1e-16 below zero, which would make a standard deviation NaN.
    aleatoric_var = np.clip(total_var - epistemic_var, 0.0, None)
    return pd.DataFrame(
        {
            "log10_value": frame[target].to_numpy(dtype=float),
            "sd_total": np.sqrt(np.clip(total_var, 0.0, None)),
            "sd_aleatoric": np.sqrt(aleatoric_var),
            "sd_epistemic": np.sqrt(epistemic_var),
        }
    )


def selftest(device: str) -> None:
    """Reproduce the repository's shipped demo prediction, which pins the log base."""
    jobs = pd.DataFrame(
        [
            {
                "SMILES": canonical_smiles(SELFTEST_SMILES, "k_cat"),
                "sequence": SELFTEST_SEQUENCE,
                "pdbpath": "seq1.pdb",
            }
        ]
    )
    with tempfile.TemporaryDirectory() as work:
        raw = _run_predict(jobs, "k_cat", device, work)
    value = float(_decompose_uncertainty(raw, TARGETS["k_cat"])["log10_value"].iloc[0])
    print(
        f"selftest log10 k_cat = {value:.10f}, "
        f"the repository's demo output records {SELFTEST_EXPECTED_LOG10:.10f}"
    )
    if abs(value - SELFTEST_EXPECTED_LOG10) > SELFTEST_TOLERANCE:
        raise RuntimeError(
            f"selftest produced {value:.10f}, not the released "
            f"{SELFTEST_EXPECTED_LOG10:.10f}. The checkpoint, the ESM-2 encoder or the "
            "output transform is wrong, and predictions written under a wrong transform "
            "are off by a power."
        )


def predict_pairs(pairs: pd.DataFrame, device: str) -> pd.DataFrame:
    """Predict both parameters for a pair table, returning value and uncertainty columns.

    Two rows of the pair table that carry the same sequence and the same canonical SMILES
    are the same prediction, so the model is run once per distinct job and the answer is
    mapped back. The GEM's pair table repeats both heavily.

    **The answer is joined back on ``(SMILES, sequence)``, never by position.** CatPred
    returns FEWER rows than it is given: a substrate with no heavy atom has no atom-bond
    graph for the D-MPNN, so the bare proton yeast-GEM writes as ``[H+]`` is dropped
    silently. Concatenating the output beside the job table by position therefore shifts
    every row after the first drop, which is a wrong number rather than a missing one.
    """
    out = pd.DataFrame(index=pairs.index)
    failures = pd.Series(pd.NA, index=pairs.index, dtype="object")
    for parameter in ("k_cat", "K_M"):
        target = TARGETS[parameter]
        canonical = {
            s: canonical_smiles(s, parameter) for s in pairs["smiles"].drop_duplicates()
        }
        keys = pd.Series([canonical[s] for s in pairs["smiles"]], index=pairs.index)
        usable = keys.notna()
        failures[~usable] = "unparsable_smiles"
        jobs = (
            pd.DataFrame(
                {"SMILES": keys[usable], "sequence": pairs.loc[usable, "sequence"]}
            )
            .drop_duplicates(["SMILES", "sequence"])
            .reset_index(drop=True)
        )
        jobs["pdbpath"] = [f"{sequence_key(s)}.pdb" for s in jobs["sequence"]]
        print(
            f"{parameter}: {len(jobs)} distinct (sequence, SMILES) jobs "
            f"for {int(usable.sum())} usable rows",
            flush=True,
        )
        with tempfile.TemporaryDirectory() as work:
            raw = _run_predict(jobs, parameter, device, work)
        scored = pd.concat(
            [raw[["SMILES", "sequence"]], _decompose_uncertainty(raw, target)], axis=1
        ).set_index(["SMILES", "sequence"])
        index = pd.MultiIndex.from_arrays([keys, pairs["sequence"]])
        for column in ("log10_value", "sd_total", "sd_aleatoric", "sd_epistemic"):
            values = scored[column].reindex(index).to_numpy()
            name = (
                parameter if column == "log10_value" else f"{parameter}_log10_{column}"
            )
            out[name] = np.power(10.0, values) if column == "log10_value" else values
        dropped = usable & out[parameter].isna()
        heavy = {s: n_heavy_atoms(s) for s in keys[dropped].drop_duplicates()}
        failures[dropped] = [
            "no_heavy_atom_substrate" if heavy[k] == 0 else "dropped_by_catpred"
            for k in keys[dropped]
        ]
    out["failure"] = failures
    return out


def main() -> None:
    """Predict both parameters over a pinned pair table and write the parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Reproduce the repository's shipped demo prediction and exit.",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest(args.device)
        return
    if not args.inputs or not args.output:
        raise ValueError(
            "--inputs and --output are required unless --selftest is given"
        )

    started = time.monotonic()
    pairs = (
        pd.read_parquet(args.inputs)
        if args.inputs.endswith(".parquet")
        else pd.read_csv(args.inputs)
    )
    if "sequence" not in pairs.columns:
        raise ValueError(
            f"{args.inputs} has no `sequence` column. CatPred is given the same protein "
            "the other predictors were given, so the sequence travels with the pair "
            "table rather than being re-derived here."
        )
    bad = [s for s in pairs["sequence"].drop_duplicates() if set(s) - VALID_RESIDUES]
    if bad:
        raise ValueError(
            f"{len(bad)} sequences carry a residue outside the 20-letter alphabet "
            "CatPred validates against; the GEM proteome carries none, so this is a "
            "disagreement between the pair table and the genome rather than a gap."
        )
    truncated = int((pairs["sequence"].str.len() > ESM_MAX_RESIDUES).sum())

    predictions = predict_pairs(pairs, args.device)
    table = pairs.join(predictions)
    table.to_parquet(args.output, index=False)

    run = CatPredRun(
        inputs_path=args.inputs,
        output_path=args.output,
        device=args.device,
        weights_sha256=anchor_weights_sha256(),
    )
    result = CatPredResult(
        run=run,
        built_at=datetime.now(UTC).isoformat(),
        runtime_seconds=round(time.monotonic() - started, 1),
        n_input_rows=int(len(pairs)),
        n_distinct_jobs=int(pairs.drop_duplicates(["sequence", "smiles"]).shape[0]),
        n_predicted={p: int(table[p].notna().sum()) for p in ("k_cat", "K_M")},
        n_failed={p: int(table[p].isna().sum()) for p in ("k_cat", "K_M")},
        failure_reasons={
            str(k): int(v) for k, v in table["failure"].value_counts().items()
        },
        value_median={
            p: float(np.nanmedian(table[p])) if table[p].notna().any() else float("nan")
            for p in ("k_cat", "K_M")
        },
        sd_total_median_decades={
            p: float(np.nanmedian(table[f"{p}_log10_sd_total"]))
            for p in ("k_cat", "K_M")
        },
        sd_aleatoric_median_decades={
            p: float(np.nanmedian(table[f"{p}_log10_sd_aleatoric"]))
            for p in ("k_cat", "K_M")
        },
        sd_epistemic_median_decades={
            p: float(np.nanmedian(table[f"{p}_log10_sd_epistemic"]))
            for p in ("k_cat", "K_M")
        },
        n_sequences_truncated=truncated,
    )
    side_car = f"{osp.splitext(args.output)[0]}_run.json"
    with open(side_car, "w") as handle:
        handle.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))
    print(f"wrote {args.output} and {side_car}")


if __name__ == "__main__":
    if PINNED_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, PINNED_SITE_PACKAGES)
    main()
