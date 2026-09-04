# experiments/026-metabolism-flux/scripts/run_eitlem_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_eitlem_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_eitlem_pinned.py

r"""Run EITLEM-Kinetics inside its pinned environment and write predictions back as parquet.

WHAT EITLEM IS
--------------
Shen et al. (2024), *Chem Catalysis*. A per-residue ESM-1v representation of the enzyme
(1,280-dim, layer 33) is attended over by a MACCS-keys fingerprint of the substrate
(167-bit), through ten ``ProMolAtt`` blocks, and a two-branch multi-head aggregator maps
the pair to one scalar. The distinctive part is the training strategy rather than the
architecture: ``k_cat`` and ``K_M`` heads are trained alternately and each iteration is
warm-started from a joint ``k_cat/K_M`` model, for eight iterations, so the two parameters
transfer into one another instead of being fit on their own sparse tables.

WHY THIS IS A SEPARATE PROCESS
------------------------------
The same shape as ``run_unikp_pinned.py``, for a different reason. EITLEM needs
``fair-esm``, which torchcell does not depend on, and it loads a 650M-parameter ESM-1v
encoder plus two predictor checkpoints. Isolating that in a subprocess keeps the encoder's
GPU allocation out of the calling process and keeps ``fair-esm`` out of the project
environment; the parquet on each side is the interface.

THE THREE CONVENTIONS THAT DECIDE WHETHER THE NUMBERS MEAN ANYTHING
--------------------------------------------------------------------
Each was read out of the authors' code rather than assumed, because each silently rescales
every value if it is guessed wrong.

* **The output is** :math:`\log_{10}`, **not** :math:`\log_2`. ``Code/dataset.py`` can
  encode targets either way, so the flag has to be traced: ``iter_train_scripts.py``
  declares ``--log10`` with ``default=True`` and the released weights are the run that
  README documents, whose worked example inverts the prediction with ``math.pow(10, res)``
  and states the answer is 1.39. Reproducing that example is the check that settles it
  (``--selftest``), and it is the check DLKcat's log2 head would have failed.
* **Units are those of the training tables**, ``Data/KCAT/kcat_data.json`` and
  ``Data/KM/km_data.json``: ``k_cat`` in 1/s (median 3.78) and ``K_M`` in mM (median
  0.227, e.g. 0.071 for NADH). Both are the BRENDA conventions and both are already the
  canonical units of ``KineticPrediction``, so nothing is rescaled here.
* **Sequences are truncated to 1,022 residues.** Not a convenience: ESM-1v's learned
  positional embedding admits 1,024 positions, and the authors' own
  ``Data/Feature/seq_str.fasta`` tops out at exactly 1,022, so every training embedding
  was computed under this cap. Anything longer is outside both the encoder and the
  training distribution, and head truncation is what ``fair-esm``'s ``extract.py`` does.

Multi-component SMILES are kept, which is where this differs from UniKP. UniKP's authors
filtered them out of their own training set, so the pipeline mirrors that filter; EITLEM's
authors did not, and 2.4% of its ``k_cat`` table and 4.3% of its ``K_M`` table contain a
``.``. Applying UniKP's filter here would drop 105 pairs that this model was in fact
trained to handle.
"""

import argparse
import functools
import hashlib
import os
import os.path as osp
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/eitlem"
CODE = osp.join(MIRROR, "Code")
EMBEDDING_CACHE = (
    "/scratch/projects/torchcell-scratch/data/torchcell/kinetics/_features/eitlem_esm1v"
)

# The final iteration of the eight-round transfer schedule, which is what the README's
# usage section names for each parameter. Earlier iterations are intermediate states of
# that schedule, not alternative models, so there is nothing to select between.
CHECKPOINTS = {
    "k_cat": "Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787",
    "K_M": "Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802",
}
MACCS_BITS = 167
ESM_MAX_RESIDUES = 1022
ESM_LAYER = 33

# The README's worked example, reproduced before any real prediction is written.
SELFTEST_SEQUENCE = (
    "MRAVRLVEIGKPLSLQEIGVPKPKGPQVLIKVEAAGVCHSDVHMRQGRFGNLRIVEDLGVKLPVTLGHEIAGKIEEVGDEV"
    "VGYSKGDLVAVNPWQGEGNCYYCRIGEEHLCDSPRWLGINFDGAYAEYVIVPHYKYMYKLRRLNAVEAAPLTCSGITTYRA"
    "VRKASLDPTKTLLVVGAGGGLGTMAVQIAKAVSGATIIGVDVREEAVEAAKRAGADYVINASMQDPLAEIRRITESKGVDA"
    "VIDLNNSEKTLSVYPKALAKQGKYVMVGLFGADLHYHAPLITLSEIQFVGSLVGNQSDFLGIMRLAEAGKVKPMITKTMKL"
    "EEANEAIDNLENFKAIGRQVLIP"
)
SELFTEST_SMILES = (
    "COC(=O)C1=CN2CCc3c([nH]c4ccccc34)[C@@]2(C)[C@@H]2CN3CCc4c([nH]c5ccccc45)"
    "[C@H]3C[C@H]12"
)
SELFTEST_EXPECTED = 1.39


def sequence_key(sequence: str) -> str:
    """Content address for a protein, so the embedding cache survives any reordering."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def maccs_keys(smiles: str) -> list[int]:
    """The 167-bit MACCS fingerprint, exactly the substrate feature ``dataset.py`` builds."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return []
    return cast(list[int], MACCSkeys.GenMACCSKeys(molecule).ToList())


def load_esm() -> tuple[Any, Any]:
    """ESM-1v (UR90S ensemble member 1), the encoder the released weights were trained on.

    ``fair-esm`` 2.0.0 pickled an ``argparse.Namespace`` into its checkpoints, which
    ``torch.load`` refuses to restore under its post-2.6 default. The file is a pinned
    artifact from the authors' own download path, so restoring it fully is the correct
    read rather than a safety hole.
    """
    import esm

    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    finally:
        torch.load = original_load
    return model, alphabet


def embed_sequences(sequences: list[str], device: str) -> None:
    """Write one per-residue ESM-1v tensor per missing sequence into the cache.

    Batch size is one, matching the README's example call, so no padding mask can leak
    into a representation.
    """
    os.makedirs(EMBEDDING_CACHE, exist_ok=True)
    missing = [
        s
        for s in sequences
        if not osp.exists(osp.join(EMBEDDING_CACHE, f"{sequence_key(s)}.pt"))
    ]
    if not missing:
        print(f"all {len(sequences)} sequence embeddings already cached", flush=True)
        return

    print(f"embedding {len(missing)} of {len(sequences)} sequences with ESM-1v", flush=True)
    model, alphabet = load_esm()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    for index, sequence in enumerate(missing):
        clipped = sequence[:ESM_MAX_RESIDUES]
        _, _, tokens = batch_converter([("protein", clipped)])
        with torch.no_grad():
            out = model(tokens.to(device), repr_layers=[ESM_LAYER], return_contacts=False)
        # Drop the BOS token and stop at the last residue, so the matrix is one row per
        # residue. This is the slice both extract.py and the README example take.
        representation = out["representations"][ESM_LAYER][0, 1 : len(clipped) + 1]
        torch.save(
            representation.cpu().clone(),
            osp.join(EMBEDDING_CACHE, f"{sequence_key(sequence)}.pt"),
        )
        if index % 100 == 0:
            print(f"  protein {index}/{len(missing)}", flush=True)

    del model
    torch.cuda.empty_cache()


def load_predictor(parameter: str, device: str) -> torch.nn.Module:
    """Instantiate one EITLEM head and load its released iteration-8 checkpoint."""
    sys.path.insert(0, CODE)
    from KCM import EitlemKcatPredictor
    from KMP import EitlemKmPredictor

    builder = EitlemKcatPredictor if parameter == "k_cat" else EitlemKmPredictor
    # The constructor arguments are fixed by the released weights, not free
    # hyperparameters: (mol_in_dim, hidden_dim, protein_dim, layer, dropout, att_layer).
    model = builder(MACCS_BITS, 512, 1280, 10, 0.5, 10)
    state = torch.load(osp.join(MIRROR, CHECKPOINTS[parameter]), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(
    model: torch.nn.Module,
    pairs: list[tuple[str, list[int]]],
    device: str,
    residues_per_batch: int,
) -> np.ndarray:
    """Run one head over (sequence, fingerprint) pairs and return values in linear space."""
    from torch_geometric.data import Batch, Data

    cache: dict[str, torch.Tensor] = {}

    def embedding(sequence: str) -> torch.Tensor:
        key = sequence_key(sequence)
        if key not in cache:
            cache[key] = torch.load(
                osp.join(EMBEDDING_CACHE, f"{key}.pt"), map_location="cpu"
            )
        return cache[key]

    values: list[float] = []
    batch: list[Data] = []
    residues = 0
    for position, (sequence, fingerprint) in enumerate(pairs):
        protein = embedding(sequence)
        batch.append(
            Data(
                x=torch.FloatTensor(fingerprint).unsqueeze(0),
                pro_emb=protein,
                num_nodes=protein.shape[0],
            )
        )
        residues += protein.shape[0]
        last = position == len(pairs) - 1
        if residues >= residues_per_batch or last:
            merged = Batch.from_data_list(batch, follow_batch=["pro_emb"]).to(device)
            with torch.no_grad():
                out = model(merged)
            values.extend(out.cpu().reshape(-1).tolist())
            batch, residues = [], 0
            print(f"  pair {len(values)}/{len(pairs)}", flush=True)
    # The head regresses log10 of the parameter; see this module's docstring for how that
    # was established and for the self-test that confirms it.
    return cast(np.ndarray, np.power(10.0, np.asarray(values, dtype=float)))


def selftest(device: str) -> None:
    """Reproduce the README's worked example, which pins the log10 convention."""
    embed_sequences([SELFTEST_SEQUENCE], device)
    model = load_predictor("k_cat", device)
    value = predict(model, [(SELFTEST_SEQUENCE, maccs_keys(SELFTEST_SMILES))], device, 1)[0]
    print(f"selftest k_cat = {value:.4f} 1/s, README states {SELFTEST_EXPECTED}")
    if abs(value - SELFTEST_EXPECTED) > 0.01:
        raise RuntimeError(
            f"selftest produced {value:.4f}, not the documented {SELFTEST_EXPECTED}. "
            "The checkpoint, the ESM-1v encoder, or the output transform is wrong, and "
            "predictions written under a wrong transform are off by a power."
        )


def main() -> None:
    """Featurize, predict both parameters, and write the results parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--residues-per-batch", type=int, default=40000)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Reproduce the README's documented example and exit.",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest(args.device)
        return

    pairs = pd.read_parquet(args.inputs)
    fingerprints = {s: maccs_keys(s) for s in pairs["smiles"].drop_duplicates()}
    usable = pairs["smiles"].map(lambda s: len(fingerprints[s]) == MACCS_BITS)
    work = pairs[usable]
    print(f"{len(work)} usable of {len(pairs)} pairs", flush=True)

    sequences = work["sequence"].drop_duplicates().tolist()
    long = sum(1 for s in sequences if len(s) > ESM_MAX_RESIDUES)
    print(
        f"{len(sequences)} distinct proteins ({long} truncated to {ESM_MAX_RESIDUES} "
        f"residues), {work['smiles'].nunique()} distinct molecules",
        flush=True,
    )
    embed_sequences(sequences, args.device)

    ordered = list(zip(work["sequence"], (fingerprints[s] for s in work["smiles"])))
    out = pairs.copy()
    for parameter in ("k_cat", "K_M"):
        model = load_predictor(parameter, args.device)
        values = predict(model, ordered, args.device, args.residues_per_batch)
        out.loc[work.index, parameter] = values
        print(f"{parameter}: median {np.median(values):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out["failure"] = [None if ok else "unparsable_smiles" for ok in usable]
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
