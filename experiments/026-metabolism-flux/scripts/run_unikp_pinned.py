# experiments/026-metabolism-flux/scripts/run_unikp_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_unikp_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_unikp_pinned.py

r"""Run UniKP inside its pinned environment and write predictions back as parquet.

WHY THIS IS A SEPARATE PROCESS
--------------------------------
UniKP's released regressors were pickled with scikit-learn before 1.3, when a
``missing_go_to_left`` field entered the decision-tree node dtype. Modern scikit-learn
refuses to unpickle them, and it is right to: silently reinterpreting a tree's node array
would give predictions that are not UniKP's. So the model runs in an environment pinned to
scikit-learn 1.2.2, and the torchcell-side dataset class calls this script rather than
importing it. The parquet on each side of the boundary is the interface.

This is the general shape for a released predictor whose dependencies conflict with the
project's, and there will be more of them: DeepEnzyme needs NVIDIA apex and a pinned old
numpy for the same reason.

WHAT UniKP IS
-------------
Yu et al. (2023), *Nature Communications*. A protein representation from ProtT5-XL-UniRef50
is concatenated with a substrate representation from a SMILES transformer, and an
ExtraTrees regressor maps the 2,048-dim concatenation to :math:`\log_{10}` of the
parameter. Three regressors were released, for :math:`k_{cat}`, :math:`K_M`, and their
ratio; the first two are used here.

Two details are copied from the authors' code because they change the features rather than
merely the plumbing: a sequence longer than 1,000 residues is truncated to its first and
last 500, and ``U``, ``Z``, ``O``, ``B`` are replaced by ``X`` before tokenization. A
multi-component SMILES is excluded, as in the authors' own training filter.
"""

import argparse
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
import torch

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/unikp"
PROT_T5 = "Rostlab/prot_t5_xl_uniref50"
MAX_SMILES_TOKENS = 218
SMILES_SEQ_LEN = 220


def sequence_key(sequence: str) -> str:
    """Stable key for a protein sequence, so the embedding cache is content-addressed."""
    import hashlib

    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def smiles_vectors(smiles_list: list[str], device: str) -> np.ndarray:
    """SMILES-transformer embeddings, 1024-dim per molecule."""
    sys.path.insert(0, MIRROR)
    from build_vocab import WordVocab
    from pretrain_trfm import TrfmSeq2seq
    from utils import split

    # vocab.pkl was pickled from a script where WordVocab lived in __main__, so the class
    # has to be visible there again or the load fails on attribute lookup. Rebuilding the
    # vocabulary instead is not an option: its token-to-index map IS part of the model.
    import __main__

    __main__.WordVocab = WordVocab
    vocab = WordVocab.load_vocab(osp.join(MIRROR, "vocab.pkl"))
    pad_index, unk_index, eos_index, sos_index = 0, 1, 2, 3

    ids_all, seg_all = [], []
    for smiles in smiles_list:
        tokens = split(smiles).split()
        if len(tokens) > MAX_SMILES_TOKENS:
            # The authors keep the two ends and drop the middle, so a long molecule still
            # contributes its termini rather than being refused.
            tokens = tokens[:109] + tokens[-109:]
        ids = [sos_index] + [vocab.stoi.get(t, unk_index) for t in tokens] + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (SMILES_SEQ_LEN - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        ids_all.append(ids)
        seg_all.append(seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(
        torch.load(osp.join(MIRROR, "trfm_12_23000.pkl"), map_location="cpu")
    )
    trfm.eval()
    return trfm.encode(torch.t(torch.tensor(ids_all)))


def sequence_vectors(sequences: list[str], device: str) -> np.ndarray:
    """ProtT5 mean-pooled residue embeddings, 1024-dim per protein."""
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(PROT_T5, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_T5).to(device).eval()

    features = np.zeros((len(sequences), 1024), dtype=float)
    for index, sequence in enumerate(sequences):
        if len(sequence) > 1000:
            sequence = sequence[:500] + sequence[-500:]
        spaced = re.sub(r"[UZOB]", "X", " ".join(sequence))
        encoded = tokenizer.batch_encode_plus(
            [spaced], add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(encoded["input_ids"]).to(device)
        attention_mask = torch.tensor(encoded["attention_mask"]).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = output.last_hidden_state.cpu().numpy()[0]
        # Drop the final token, which is the end-of-sequence marker rather than a residue.
        length = int((attention_mask[0] == 1).sum()) - 1
        features[index] = embedding[:length].mean(axis=0)
        if index % 100 == 0:
            print(f"  protein {index}/{len(sequences)}", flush=True)
    return features


def main() -> None:
    """Featurize, predict both parameters, and write the results parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--protein-embeddings",
        default=None,
        help="npz of precomputed ProtT5 vectors keyed by sequence hash. ProtT5-XL is a "
        "3B-parameter encoder and this environment pins CPU torch, so the embeddings are "
        "produced once on GPU by embed_prot_t5.py and reused.",
    )
    args = parser.parse_args()

    import pickle
    import warnings

    warnings.filterwarnings("ignore")

    pairs = pd.read_parquet(args.inputs)
    usable = pairs["smiles"].notna() & ~pairs["smiles"].astype(str).str.contains(
        r"\.", regex=True
    )
    work = pairs[usable]
    print(f"{len(work)} usable of {len(pairs)} pairs", flush=True)

    # Featurize each DISTINCT protein and molecule once. The pair table repeats a sequence
    # for every substrate it acts on, so featurizing per row would run ProtT5 roughly six
    # times more than necessary.
    sequences = work["sequence"].drop_duplicates().tolist()
    smiles = work["smiles"].drop_duplicates().tolist()
    print(f"{len(sequences)} distinct proteins, {len(smiles)} distinct molecules", flush=True)

    if args.protein_embeddings:
        cached = np.load(args.protein_embeddings, allow_pickle=True)
        by_hash = dict(zip(cached["keys"].tolist(), cached["vectors"]))
        missing = [s for s in sequences if sequence_key(s) not in by_hash]
        if missing:
            raise KeyError(
                f"{len(missing)} sequences absent from {args.protein_embeddings}. "
                "Rerun embed_prot_t5.py; a partial cache would silently drop enzymes."
            )
        seq_matrix = np.stack([by_hash[sequence_key(s)] for s in sequences])
    else:
        seq_matrix = sequence_vectors(sequences, args.device)
    seq_index = {s: i for i, s in enumerate(sequences)}
    smi_matrix = np.asarray(smiles_vectors(smiles, args.device))
    smi_index = {s: i for i, s in enumerate(smiles)}

    features = np.concatenate(
        [
            smi_matrix[[smi_index[s] for s in work["smiles"]]],
            seq_matrix[[seq_index[s] for s in work["sequence"]]],
        ],
        axis=1,
    )
    print(f"feature matrix {features.shape}", flush=True)

    out = pairs.copy()
    for parameter, filename in (("k_cat", "UniKP for kcat.pkl"), ("K_M", "UniKP for Km.pkl")):
        with open(osp.join(MIRROR, filename), "rb") as handle:
            regressor = pickle.load(handle)
        log10_value = regressor.predict(features)
        out.loc[work.index, parameter] = np.power(10.0, log10_value)
        print(f"{parameter}: median {np.median(np.power(10.0, log10_value)):.3f}", flush=True)

    out["failure"] = np.where(usable, None, "multi_component_smiles")
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
