---
id: 0tz1i2sh3a31y4qnwny02cc
title: Encoders
desc: ''
updated: 1790380023196
created: 1790380023196
---

## 2026.09.25 - Small-molecule encoder registry

`torchcell/molecule/` turns a SMILES list into one fixed-width float32 vector per
molecule. Every encoder is a `MoleculeEncoder` subclass (`name`, `dim`,
`check(smiles) -> Mol`, `encode(list[str]) -> (n, dim)`) registered in
`torchcell.molecule.ENCODERS`. `standardize(smiles)` returns RDKit's canonical isomeric
SMILES with charges, stereo and every fragment kept (no neutralization or salt
stripping). Contract, enforced by `tests/torchcell/molecule/test_encoders.py`:

- An unparsable SMILES raises `ValueError` before any model runs; no encoder returns a
  zero row for a molecule it could not read. The caller owns coverage accounting.
- Deterministic: eval mode, `inference_mode`, MoLFormer with `deterministic_eval=True`,
  Uni-Mol conformers at unimol_tools' fixed seed 42. Two calls agree to 1e-5.
- Spelling-independent: fingerprints act on the parsed mol; text encoders are fed the
  canonical SMILES, so `OCC` and `CCO` embed identically.
- `rdkit_2d` is the one encoder that may emit NaN (undefined or overflowed descriptor);
  the caller imputes. `feature_names` lists the 217 columns in `Descriptors.descList`
  order.
- `unimol_v1.check` raises for a molecule unimol_tools would embed from stand-in
  coordinates. unimol_tools' `inner_smi2coords` quietly substitutes flat 2D coordinates
  when ETKDG fails (metal dihalides, nitric oxide, bleomycin at 180 atoms) and all-zero
  coordinates for isolated ions (`[Cl-].[Na+]`), then embeds them as if 3D; on the
  031 union that was 11 of 343 compounds embedded from nothing.

Weights not on the Hugging Face hub (Mol2Vec's gensim pickle, MolE's Zenodo checkpoint)
live at `$DATA_ROOT/data/torchcell/molecule_encoders/<name>/` beside a pydantic
`manifest.json` (`torchcell.molecule.weights.WeightsManifest`: name, source_url,
retrieval_method, retrieval_command, sha256, retrieved_at, bytes). The sha256 is pinned
in `WEIGHTS`; `ensure_weights(name)` downloads once with the recorded curl command and
re-hashes on every load, raising on drift or on a file with no manifest. HF models cache
in the default hub cache under their revision hash.

`torchcell/molecule/similarity.py`: `tanimoto_matrix(a, b)` is the generalized
(min/max) Tanimoto, equal to the bit Tanimoto on 0/1 vectors and to RDKit's
`TanimotoSimilarity` on both bit and count Morgan fingerprints (asserted in
`test_similarity.py`); `cosine_matrix(a, b)`. A zero denominator reports 0.0.

### Encoder table

`s / 100` is seconds per 100 molecules on GilaHyper (RTX 6000 Ada, cuda:0 shared with
other jobs; RDKit on one CPU core), measured over the 343-compound union of the three
env-chemgen datasets by
`experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_compounds.py` and written
to `experiments/031-env-chemgen-inhibitor-tolerance/results/encoder_timing.md` (run of
2026.09.25; model load time excluded, listed there separately).

| encoder | input | dim | source / weights | installed | s / 100 |
|---|---|---|---|---|---|
| `ecfp4_count` | mol | 2048 | RDKit `GetMorganGenerator(radius=2, fpSize=2048)`, counts | yes | 0.021 |
| `ecfp4_bit` | mol | 2048 | same generator, bit vector | yes | 0.020 |
| `fcfp4_count` | mol | 2048 | Morgan radius 2 with `GetMorganFeatureAtomInvGen()`, counts | yes | 0.030 |
| `maccs` | mol | 167 | RDKit `MACCSkeys.GenMACCSKeys` | yes | 0.077 |
| `rdkit_2d` | mol | 217 | RDKit `Descriptors.CalcMolDescriptors` (RDKit 2026.03.6) | yes | 0.609 |
| `mol2vec` | mol, Morgan ids r0+r1 | 300 | samoturk/mol2vec `model_300dim.pkl`, sha256 `62934b4e...`, gensim 4.4.0 loads the gensim-3 pickle without a compatibility shim | yes | 0.031 |
| `chemberta2_mlm` | canonical SMILES | 384 | HF `DeepChem/ChemBERTa-77M-MLM`, mean over non-pad tokens | yes | 0.097 |
| `chemberta2_mtr` | canonical SMILES | 384 | HF `DeepChem/ChemBERTa-77M-MTR`, mean pool | yes | 0.054 |
| `molformer_xl` | canonical SMILES | 768 | HF `ibm-research/MoLFormer-XL-both-10pct` pinned to revision `7b12d946` (2024-03-31); the 2026-07 revisions import `create_bidirectional_mask`, absent in transformers 4.57.1 | yes | 0.158 |
| `roberta_zinc_480m` | canonical SMILES | 768 | HF `entropy/roberta_zinc_480m`, mean pool, width from config | yes | 0.188 |
| `unimol_v1` | canonical SMILES -> ETKDG conformer | 512 | `unimol_tools` 0.1.6 `UniMolRepr(data_type="molecule", remove_hs=False)`, CLS token; checkpoint `mol_pre_all_h_220816.pt` fetched from HF `dptech/Uni-Mol-Models`; installing it downgraded numpy 2.3.4 -> 2.2.6 | yes | 15.2 |
| `mole_static` | mol with explicit H -> PyG graph | 1000 | MolE `gin_concat_R1000_E8000_lambda0.0001`, Zenodo 10803099 `model.pth` (804 MB, sha256 `2d324644...`), model code vendored under `torchcell/molecule/third_party/mole/` (MIT) | yes | 0.075 |

Sanity check from the tests (cosine similarity, ethanol vs acetic acid and ethanol vs
vanillin): every encoder puts ethanol closer to acetic acid, asserted for the four
fingerprints and only printed for the rest. `molformer_xl` gives 0.100 vs 0.016,
`chemberta2_mtr` 0.721 vs 0.028, `mol2vec` 0.776 vs 0.688, `unimol_v1` 0.787 vs 0.694.
