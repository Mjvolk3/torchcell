# tests/torchcell/models/test_models_genslm.py
# [[tests.torchcell.models.test_models_genslm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_models_genslm.py
"""``GenSLM`` on the vendored codon tokenizer and a two-layer, 16-wide GPT-NeoX built
in the test, saved as a Lightning-style checkpoint under ``tmp_path`` with a weights
manifest; no released checkpoint is read.

Contract: codons are the tokens (69-token vocabulary: 64 codons plus ``[UNK] [CLS]
[SEP] [PAD] [MASK]``), the context is 2,048 codons, an out-of-frame or non-ACGT
sequence is refused, the checkpoint is sha256-verified against the manifest before
loading, a checkpoint missing a parameter is refused, and the mean embedding is the
mean over real codons only (pad positions excluded).
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord, sha256_file
from torchcell.models import genslm as genslm_module
from torchcell.models.genslm import (
    GENSLM_MODELS,
    GENSLM_SEQ_LENGTH,
    GenSLM,
    GenSLMModelSpec,
    GenSLMWeightsManifest,
    codon_tokens,
)

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
TINY_ID = "genslm_tiny_test"
TINY_WEIGHTS = "tiny.pt"


def _tiny_config(path: Path) -> Path:
    """A 16-wide, two-layer GPT-NeoX over the 69-token vocabulary, as a JSON file."""
    config = GPTNeoXConfig(
        vocab_size=69,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=GENSLM_SEQ_LENGTH,
        rotary_pct=0.25,
        use_cache=False,
    )
    config_path = path / "neox_tiny.json"
    config_path.write_text(config.to_json_string())
    return config_path


def _write_manifest(weights_dir: Path, weights_file: str) -> None:
    manifest = GenSLMWeightsManifest(
        files={
            weights_file: RetrievalRecord(
                method=RetrievalMethod.globus,
                source_url=f"globus://{genslm_module.GLOBUS_ENDPOINT_ID}/{weights_file}",
                retriever="test",
                sha256=sha256_file(weights_dir / weights_file),
                retrieved_at="2026-10-07",
            )
        }
    )
    (weights_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


@pytest.fixture
def tiny_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenSLM:
    """``GenSLM`` over a seeded tiny model saved as ``{"state_dict": ...}`` + manifest."""
    torch.manual_seed(0)
    config_path = _tiny_config(tmp_path)
    reference = GPTNeoXForCausalLM(GPTNeoXConfig.from_json_file(str(config_path)))
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save({"state_dict": reference.state_dict()}, weights_dir / TINY_WEIGHTS)
    _write_manifest(weights_dir, TINY_WEIGHTS)
    spec = GenSLMModelSpec(
        model_id=TINY_ID,
        config_asset=str(config_path),
        weights_file=TINY_WEIGHTS,
        hidden_size=16,
    )
    monkeypatch.setitem(GENSLM_MODELS, TINY_ID, spec)
    monkeypatch.setattr(GenSLM, "VALID_MODEL_NAMES", list(GENSLM_MODELS))
    return GenSLM(TINY_ID, weights_dir=str(weights_dir))


# ---- tokenizer ------------------------------------------------------------------ #


def test_vocabulary_is_64_codons_plus_5_specials() -> None:
    tokenizer = GenSLM.build_tokenizer()
    vocab = tokenizer.get_vocab()
    assert len(vocab) == 69
    assert all(codon in vocab for codon in CODONS)
    assert set(vocab) - set(CODONS) == {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}
    assert tokenizer.pad_token == "[PAD]"


def test_codon_tokens_groups_in_frame_and_uppercases() -> None:
    assert codon_tokens("atgGCCtaa") == "ATG GCC TAA"


def test_codon_tokens_refuses_out_of_frame_length() -> None:
    with pytest.raises(ValueError, match="not a multiple of 3"):
        codon_tokens("ATGGCCTAAG")


def test_codon_tokens_refuses_non_acgt() -> None:
    with pytest.raises(ValueError, match=r"non-ACGT characters \['N'\]"):
        codon_tokens("ATGNNNTAA")


def test_tokenize_truncates_to_context_and_pads_to_longest(tiny_model: GenSLM) -> None:
    long = "ATG" * (GENSLM_SEQ_LENGTH + 10)
    batch = tiny_model.tokenize([long, "ATGGCCTAA"])
    assert batch["input_ids"].shape == (2, GENSLM_SEQ_LENGTH)
    assert batch["attention_mask"][0].sum().item() == GENSLM_SEQ_LENGTH
    assert batch["attention_mask"][1].sum().item() == 3
    pad_id = tiny_model.tokenizer.pad_token_id
    assert batch["input_ids"][1, 3:].eq(pad_id).all()
    vocab = tiny_model.tokenizer.get_vocab()
    assert batch["input_ids"][1, :3].tolist() == [
        vocab["ATG"],
        vocab["GCC"],
        vocab["TAA"],
    ]


# ---- loading -------------------------------------------------------------------- #


def test_registry_lists_the_four_foundation_models() -> None:
    released = {k: v for k, v in GENSLM_MODELS.items() if k != TINY_ID}
    assert list(released) == [
        "genslm_25M_patric",
        "genslm_250M_patric",
        "genslm_2.5B_patric",
        "genslm_25B_patric",
    ]
    assert [s.hidden_size for s in released.values()] == [512, 1840, 3840, 8196]
    for spec in released.values():
        config = json.loads(
            (Path(genslm_module.ASSETS_DIR) / spec.config_asset).read_text()
        )
        assert config["hidden_size"] == spec.hidden_size
        assert config["vocab_size"] == 69
        assert config["max_position_embeddings"] == GENSLM_SEQ_LENGTH


def test_vendored_assets_match_their_provenance_pins() -> None:
    assets = Path(genslm_module.ASSETS_DIR)
    provenance = json.loads((assets / "provenance.json").read_text())
    assert provenance["license"] == "MIT"
    for record in provenance["files"]:
        assert sha256_file(assets / record["path"]) == record["sha256"]


def test_unknown_model_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid model_name 'genslm_nope'"):
        GenSLM("genslm_nope", weights_dir=str(tmp_path))


def test_missing_manifest_names_globus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Globus"):
        GenSLM("genslm_25M_patric", weights_dir=str(tmp_path))


def test_sha256_mismatch_is_refused(tiny_model: GenSLM) -> None:
    path = Path(tiny_model.weights_path)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="manifest pins"):
        tiny_model.verify_weights()


def test_checkpoint_missing_a_parameter_is_refused(tiny_model: GenSLM) -> None:
    checkpoint = torch.load(tiny_model.weights_path, weights_only=False)
    del checkpoint["state_dict"]["gpt_neox.layers.0.mlp.dense_h_to_4h.weight"]
    torch.save(checkpoint, tiny_model.weights_path)
    _write_manifest(Path(tiny_model.weights_dir), TINY_WEIGHTS)
    with pytest.raises(ValueError, match="missing \\['gpt_neox.layers.0.mlp"):
        GenSLM(TINY_ID, weights_dir=tiny_model.weights_dir)


def test_checkpoint_with_unexpected_key_is_refused(tiny_model: GenSLM) -> None:
    checkpoint = torch.load(tiny_model.weights_path, weights_only=False)
    checkpoint["state_dict"]["extra.weight"] = torch.zeros(1)
    torch.save(checkpoint, tiny_model.weights_path)
    _write_manifest(Path(tiny_model.weights_dir), TINY_WEIGHTS)
    with pytest.raises(ValueError, match="unexpected \\['extra.weight'\\]"):
        GenSLM(TINY_ID, weights_dir=tiny_model.weights_dir)


def test_loaded_weights_equal_the_checkpoint(tiny_model: GenSLM) -> None:
    checkpoint = torch.load(tiny_model.weights_path, weights_only=False)
    loaded = tiny_model.model.state_dict()
    for name, tensor in checkpoint["state_dict"].items():
        assert torch.equal(loaded[name].cpu(), tensor), name


# ---- embedding ------------------------------------------------------------------ #


def test_embed_shapes(tiny_model: GenSLM) -> None:
    per_codon = tiny_model.embed(["ATGGCCTAA", "ATGTAA"])
    assert per_codon.shape == (2, 3, 16)
    mean = tiny_model.embed(["ATGGCCTAA", "ATGTAA"], mean_embedding=True)
    assert mean.shape == (2, 16)
    assert tiny_model.hidden_size == 16
    assert tiny_model.max_sequence_size == GENSLM_SEQ_LENGTH


def test_mean_embedding_ignores_pad_positions(tiny_model: GenSLM) -> None:
    short = "ATGGCCTAA"
    alone = tiny_model.embed(short, mean_embedding=True)[0]
    padded = tiny_model.embed(["ATG" * 50, short], mean_embedding=True)[1]
    assert torch.allclose(alone, padded, atol=1e-5)
    per_codon = tiny_model.embed(short)[0]
    assert torch.allclose(alone, per_codon.mean(dim=0), atol=1e-6)
