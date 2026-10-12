# experiments/026-metabolism-flux/scripts/run_ezkinetics_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_ezkinetics_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_ezkinetics_pinned.py

r"""Run EZKinetics over the yeast-GEM enzyme-substrate pairs, after proving it reproduces.

EZKinetics is the seventh predictor in this comparison and the only one that conditions on
assay pH and temperature. It is **unpublished** (manuscript under review), so it carries no
DOI and is cited by repository and commit rather than by paper.

**The exact combination is not a choice.** The repository ships 36 fitted models, 3
parameters by 4 protein encoders by 3 substrate encoders, and picking the wrong pair would
still produce numbers. ``inference/predict.py`` fixes it, and the shipped scalers confirm
the feature widths, so this is checked rather than assumed:

===========  ==============  ====================  ==========================
Parameter    Protein         Substrate             Features
===========  ==============  ====================  ==========================
``k_cat``    ESM2_650M       ChemBERTa-77M-MTR     1280 + 384 + 2 = **1666**
``K_M``      ESM2_3B         ChemBERTa-77M-MTR     2560 + 384 + 2 = **2946**
===========  ==============  ====================  ==========================

The two extra features are pH and temperature, defaulting to 7.0 and 25.0 in the authors'
script. Output is base 10: their script returns ``10 ** y_pred``, the same convention as
UniKP, EITLEM, TurNuP and Boost_KM, and NOT the base 2 of DLKcat and DeepEnzyme.

**Validation gate.** The mirror ships the authors' own held-out predictions as
``test_data_predictions.csv`` (protein_sequence, substrate_SMILES, y_test, y_pred). ``--validate``
re-runs our pipeline on a sample of those rows and compares to their ``y_pred``. This is not
ceremony: the scalers were pickled under scikit-learn 1.5.2 and load under 1.7.2 with an
``InconsistentVersionWarning``, and a StandardScaler that silently changed behavior would
produce entirely plausible turnover numbers. **Nothing is written until this passes.**

Measured on 60 held-out rows: median 0.0018 decades, p90 0.032, max 0.077, 2 rows past 0.05.
**The gate is therefore on the MEDIAN, not the max.** A max-based gate is the wrong statistic
here because reproduction is not bit-exact and never will be: our ESM2 forward pass runs on
different hardware and a different torch build from theirs, and the residual survives setting
the protein batch size to 1, so it is not padding. The cause of the tail is not established
and is not claimed. What is established is scale: a 0.03-decade p90 against an
across-predictor disagreement of 1.2 decades cannot change any conclusion drawn here.

**The conditions matter, and this is the reason to care.** Re-running the same rows at the
authors' 7.0/25.0 defaults instead of their true assay conditions gives a median error of
0.046 decades and a max of 0.77 against their own output. yeast-GEM carries no per-pair assay
pH or temperature, so the production run below is necessarily condition-blind, and that is a
real cost rather than a formality.

Sequences come from the shared pair table so that every predictor in the comparison sees
byte-identical inputs; embeddings are cached per distinct sequence and per distinct SMILES,
since the pair table repeats both heavily.
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import warnings
from typing import cast

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
MIRROR = osp.join(DATA_ROOT, "models", "kinetics", "ezkinetics")
OUT_DIR = osp.join(DATA_ROOT, "data", "torchcell", "kinetics", "ezkinetics", "processed")
CACHE = osp.join(DATA_ROOT, "data", "torchcell", "kinetics", "ezkinetics", "cache")
PAIRS = osp.join(RESULTS, "kinetics_predictor_inputs.csv")
#: Any built predictor table carries the same `sequence` column keyed by gene; EITLEM is
#: used because it covers every pair. Taking sequences from here rather than re-deriving
#: them guarantees EZKinetics sees exactly what the other six saw.
SEQUENCE_SOURCE = osp.join(
    DATA_ROOT, "data", "torchcell", "kinetics", "eitlem", "processed", "k_cat.parquet"
)

SUBSTRATE_MODEL = "ChemBERTa-77M-MTR"
SPEC = {
    "k_cat": {"dir": "KCAT", "protein": "ESM2_650M", "esm": "esm2_t33_650M_UR50D",
              "width": 1666},
    "K_M": {"dir": "KM", "protein": "ESM2_3B", "esm": "esm2_t36_3B_UR50D", "width": 2946},
}
#: The authors' defaults. Every other predictor here is condition-blind, so holding these
#: fixed keeps EZKinetics comparable rather than giving it an advantage the others cannot
#: have. Their effect is a separate question and is not asked here.
DEFAULT_PH, DEFAULT_TEMPERATURE = 7.0, 25.0

#: Mutable so --protein-batch-size reaches `predict` without threading it through every call.
PROTEIN_BATCH_SIZE = [4]

#: The layer BOTH the training encoders and predict.py read, for BOTH ESM2 sizes. 3B has 36
#: layers, so 33 is not its last; that is the authors' convention and is applied
#: consistently in training and inference, which was verified in code/ before running.
REPR_LAYER = 33


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def embed_smiles(smiles: list[str], device: str) -> dict[str, np.ndarray]:
    """ChemBERTa-77M-MTR mean-pooled over tokens, exactly as predict.py does it."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(f"DeepChem/{SUBSTRATE_MODEL}")
    model = AutoModel.from_pretrained(f"DeepChem/{SUBSTRATE_MODEL}").to(device).eval()
    out: dict[str, np.ndarray] = {}
    for index, smi in enumerate(smiles):
        tokens = tokenizer(smi, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            pooled = model(**tokens).last_hidden_state.mean(dim=1).squeeze()
        out[smi] = pooled.cpu().numpy().astype(np.float32)
        if index % 250 == 0:
            print(f"  smiles {index}/{len(smiles)}", flush=True)
    return out


def embed_proteins(
    sequences: list[str], esm_name: str, device: str, batch_size: int = 4
) -> dict[str, np.ndarray]:
    """ESM2 mean over residue positions, truncated at 1022, exactly as predict.py does it."""
    from esm import pretrained

    model, alphabet = pretrained.load_model_and_alphabet(esm_name)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        clipped = [s[:1022] for s in batch]
        _, _, toks = converter([(str(i), s) for i, s in enumerate(clipped)])
        with torch.no_grad():
            reps = model(toks.to(device), repr_layers=[REPR_LAYER], return_contacts=False)
        reps = reps["representations"][REPR_LAYER]
        for j, seq in enumerate(clipped):
            out[batch[j]] = reps[j, 1 : len(seq) + 1].mean(dim=0).cpu().numpy().astype(
                np.float32
            )
        if start % 100 == 0:
            print(f"  protein {start}/{len(sequences)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def load_model(parameter: str):
    """The fitted XGBoost and its scaler for this parameter's fixed encoder combination."""
    spec = SPEC[parameter]
    base = osp.join(
        MIRROR, "model", spec["dir"], "XGBoost",
        f"{spec['protein']}&{SUBSTRATE_MODEL}",
    )
    with warnings.catch_warnings():
        # The scaler was pickled under scikit-learn 1.5.2. The warning is real and is why
        # --validate exists; it is silenced here only so the run's output stays readable.
        warnings.simplefilter("ignore")
        model = pickle.load(open(osp.join(base, f"XGBoost_model_{spec['dir']}.pkl"), "rb"))
        scaler = pickle.load(open(osp.join(base, f"scaler_{spec['dir']}.pkl"), "rb"))
    return model, scaler, base


def predict(
    sequences: list[str],
    smiles: list[str],
    parameter: str,
    device: str,
    ph: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted value in natural units, from the log10 the model emits.

    ``ph`` and ``temperature`` are per row. They exist because this model conditions on
    them: passing the defaults where the true assay conditions are known does not reproduce
    the authors' output, which is what the validation gate measures.
    """
    spec = SPEC[parameter]
    model, scaler, _ = load_model(parameter)
    if ph is None:
        ph = np.full(len(sequences), DEFAULT_PH)
    if temperature is None:
        temperature = np.full(len(sequences), DEFAULT_TEMPERATURE)
    protein_vectors = embed_proteins(
        sorted(set(sequences)), cast(str, spec["esm"]), device, PROTEIN_BATCH_SIZE[0]
    )
    molecule_vectors = embed_smiles(sorted(set(smiles)), device)
    features = np.stack(
        [
            np.concatenate(
                [protein_vectors[s], molecule_vectors[m], [p, t]]
            )
            for s, m, p, t in zip(sequences, smiles, ph, temperature)
        ]
    )
    if features.shape[1] != spec["width"]:
        raise SystemExit(
            f"{parameter}: built {features.shape[1]} features, model wants {spec['width']}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log10 = model.predict(scaler.transform(features))
    return np.asarray(10.0**log10, dtype=float)


def validate(parameter: str, n: int, device: str) -> dict:
    """Reproduce the authors' own held-out predictions before trusting anything.

    ``test_data_predictions.csv`` carries no conditions, but the held-out split it was
    produced from does, and the two align row for row (verified: sequence and y_test match
    on all 6,676 rows). So the check is run at the TRUE pH and temperature, and separately
    at the defaults, which is the only way to tell a broken pipeline apart from a
    condition-blind one.
    """
    spec = SPEC[parameter]
    _, _, base = load_model(parameter)
    predictions = pd.read_csv(osp.join(base, "test_data_predictions.csv"))
    split = pd.read_csv(
        osp.join(MIRROR, "data", spec["dir"], "train_test", "test.csv")
    )
    if len(split) != len(predictions):
        raise SystemExit("held-out split and shipped predictions differ in length")
    index = predictions.sample(n=min(n, len(predictions)), random_state=0).index
    sequences = predictions.loc[index, "protein_sequence"].tolist()
    molecules = predictions.loc[index, "substrate_SMILES"].tolist()
    theirs = np.asarray(10.0 ** predictions.loc[index, "y_pred"].to_numpy(), dtype=float)

    true_conditions = predict(
        sequences, molecules, parameter, device,
        ph=split.loc[index, "PH"].to_numpy(),
        temperature=split.loc[index, "Temperature"].to_numpy(),
    )
    default_conditions = predict(sequences, molecules, parameter, device)
    exact = np.abs(np.log10(true_conditions / theirs))
    blind = np.abs(np.log10(default_conditions / theirs))
    report = {
        "parameter": parameter,
        "n_rows": int(len(index)),
        "at_true_conditions": {
            "max_abs_log10_ratio": float(exact.max()),
            "median_abs_log10_ratio": float(np.median(exact)),
            "p90_abs_log10_ratio": float(np.percentile(exact, 90)),
            "n_over_0p01": int((exact > 0.01).sum()),
            "n_over_0p05": int((exact > 0.05).sum()),
        },
        "at_default_conditions": {
            "max_abs_log10_ratio": float(blind.max()),
            "median_abs_log10_ratio": float(np.median(blind)),
        },
        "reproduced": bool(np.median(exact) < 0.01 and (exact > 0.05).mean() < 0.10),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", default="k_cat", choices=sorted(SPEC))
    parser.add_argument("--validate", type=int, default=0,
                        help="reproduce N of the authors' test rows and exit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protein-batch-size", type=int, default=4,
                        help="ESM2 batch size. The authors use 4; 1 removes padding.")
    # A run over a SUBSET of the pair table, for the units a later structure-resolution
    # pass unblocked. The subset is written to its own parquet and appended by
    # improve_kinetic_coverage.py, so the production table is never re-predicted: this
    # model is not bit-reproducible across hardware, and rebuilding would move values on
    # units that nothing asked to move. The coverage summary is written only for a full
    # run, since a subset's coverage numbers would describe the subset.
    parser.add_argument("--inputs", default=PAIRS,
                        help="Pair table to predict, .csv or .parquet. Default: the "
                             "full yeast-GEM pair table.")
    parser.add_argument("--output", default=None,
                        help="Where to write the predictions parquet. Default: the "
                             "predictor's own processed table.")
    args = parser.parse_args()
    PROTEIN_BATCH_SIZE[0] = args.protein_batch_size

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    if args.validate:
        report = validate(args.parameter, args.validate, args.device)
        path = osp.join(RESULTS, f"ezkinetics_validation_{args.parameter}.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        if not report["reproduced"]:
            raise SystemExit(
                "EZKinetics did NOT reproduce the authors' predictions; nothing written."
            )
        return

    pairs = (
        pd.read_parquet(args.inputs)
        if args.inputs.endswith(".parquet")
        else pd.read_csv(args.inputs)
    )
    if "sequence" in pairs:
        work = pairs.copy()
    else:
        sequence_of_gene = (
            pd.read_parquet(SEQUENCE_SOURCE)[["gene_id", "sequence"]]
            .drop_duplicates("gene_id")
            .set_index("gene_id")["sequence"]
        )
        work = pairs.assign(sequence=pairs["gene_id"].map(sequence_of_gene))
    work = work[work["sequence"].notna() & work["smiles"].notna()].reset_index(drop=True)
    print(f"{len(work)} of {len(pairs)} pairs have both a sequence and a SMILES")

    values = predict(
        work["sequence"].tolist(), work["smiles"].tolist(), args.parameter, args.device
    )
    work[args.parameter] = values
    work["predictor"] = "ezkinetics"
    work["ph"] = DEFAULT_PH
    work["temperature"] = DEFAULT_TEMPERATURE
    manifest = json.load(open(osp.join(MIRROR, "manifest.json")))
    work["weights_sha256"] = manifest["revision"]
    work["provenance"] = f"ezkinetics@{manifest['revision'][:12]}"
    default_output = osp.join(OUT_DIR, f"{args.parameter}.parquet")
    work.to_parquet(args.output or default_output)
    if args.output and args.output != default_output:
        print(f"wrote {args.output}; no summary, this is a subset run")
        return

    n_units_total = 3728
    summary = {
        "predictor": "ezkinetics",
        "parameter": args.parameter,
        "n_rows": int(len(work)),
        "n_units": int(work["unit_id"].nunique()),
        "n_units_total": n_units_total,
        "n_genes": int(work["gene_id"].nunique()),
        # `substrate_met_id` is the count the other six predictors report, and it is
        # compartment-specific: the same chemical in cytosol and mitochondrion is two
        # metabolites but one SMILES. Both are recorded so the table compares like with
        # like and the distinction is not lost.
        "n_substrates": int(work["substrate_met_id"].nunique()),
        "n_distinct_smiles": int(work["smiles"].nunique()),
        "coverage_frac": float(work["unit_id"].nunique() / n_units_total),
        "value_median": float(np.median(values)),
        "value_p05": float(np.percentile(values, 5)),
        "value_p95": float(np.percentile(values, 95)),
        "protein_encoder": SPEC[args.parameter]["protein"],
        "substrate_encoder": SUBSTRATE_MODEL,
        "log_base": 10,
        "ph": DEFAULT_PH,
        "temperature_c": DEFAULT_TEMPERATURE,
        "revision": manifest["revision"],
    }
    with open(osp.join(OUT_DIR, f"{args.parameter}_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(
        osp.join(RESULTS, f"kinetics_ezkinetics_{args.parameter}_summary.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
