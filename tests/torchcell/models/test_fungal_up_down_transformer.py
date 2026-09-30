"""Tests for the FungalUpDownTransformer embedding model.

2026.09.30, Phase 18. The original class below loads the real SpeciesLM weights and is
network-gated. The hermetic tests after it replace the module-level ``AutoTokenizer``
and ``AutoModelForMaskedLM`` with recorders, point the module's ``__file__`` into
``tmp_path`` (the download check creates its cache directory beside the source file)
and set ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``, so nothing reaches the Hub.

Fake tokenizer: splits the text on spaces and returns ``[0] + [3 + i for each word i]
+ [1]`` (a CLS, one id per word, a SEP) with ``token_type_ids`` all 0 and
``attention_mask`` all 1. A sequence of L bases gives L - 5 six-mers, so the token
count is 1 (CLS) + 1 (species) + (L - 5) + 1 (SEP) = L - 2; a 1003 bp upstream
sequence gives exactly 1001 tokens, the length the wrapper pads to.

Fake model: 13 hidden states (embedding layer plus 12 blocks, the size the network
test's refusal "Max layer is 13" implies) with ``hidden_states[l][0, t] = [t, l]``.
So a pooled embedding's first coordinate is the mean token position kept and its
second is the mean layer index averaged. Worked values used below:

* ``target_layer=(8,)`` averages layers 8 to 12, second coordinate 10.
* An 11 bp downstream sequence: 9 tokens, mean position (0 + ... + 8) / 9 = 4.
* A 12 bp downstream sequence: 10 tokens, mean position 4.5.
* A 10 bp upstream sequence: 8 tokens, padded with 1001 - 8 = 993 zeros after
  position 1, so ``pad_start = 2``, ``pad_end = 995``; the kept rows are positions
  0, 1 and 995 to 1000, whose mean is (0 + 1 + 995 + 996 + 997 + 998 + 999 + 1000)
  / 8 = 5986 / 8 = 748.25 (the mean over all 1001 rows would be 500).
"""

# tests/torchcell/models/test_fungal_up_down_transformer.py
# [[tests.torchcell.models.test_fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_fungal_up_down_transformer.py

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

import torchcell.models.fungal_up_down_transformer as fudt_module
from torchcell.models.fungal_up_down_transformer import FungalUpDownTransformer


# ``from_pretrained`` reaches the HuggingFace Hub API (repo template listing) even when
# the weights are cached, and a fresh runner downloads them; this class is network-gated
# and runs only with --network. The hermetic tests below it fake the backbone.
@pytest.mark.network
class TestFungalUpDownTransformerUpstream:
    """Tests for the upstream-species variant of the transformer."""

    @pytest.fixture
    def model(self):
        """Return an upstream-species model targeting layer 8."""
        return FungalUpDownTransformer(
            model_name="upstream_species_lm", target_layer=(8,)
        )

    def test_init(self, model):
        """Verify the model stores the target layer, name, and HF model dir."""
        assert model.target_layer == (8,)
        assert model.model_name == "upstream_species_lm"
        assert model.hugging_model_dir == "gagneurlab/SpeciesLM"

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_check_and_download_model_exists(self, makedirs_mock, exists_mock, model):
        """Verify no directories are created when the model already exists."""
        exists_mock.return_value = True
        model._check_and_download_model()
        makedirs_mock.assert_not_called()

    def test_max_sequence_size_upstream(self, model):
        """Verify the upstream model reports a max sequence size of 1003."""
        model.model_name = "upstream_species_lm"
        assert model.max_sequence_size == 1003

    def test_max_sequence_size_downstream(self, model):
        """Verify the downstream model reports a max sequence size of 300."""
        model.model_name = "downstream_species_lm"
        assert model.max_sequence_size == 300

    def test_pad_sequence(self, model):
        """Verify padding extends a tokenized sequence by the expected amount."""
        # Mock a tokenized_data
        tokenized_data = {
            "input_ids": torch.Tensor([[1]]),
            "token_type_ids": torch.Tensor([[1]]),
            "attention_mask": torch.Tensor([[1]]),
        }
        result, pad_start, pad_end = model._pad_sequence(
            tokenized_data, mean_embedding=True
        )

        assert pad_end - pad_start == 1000

    def test_embed_raises_value_error_for_upstream(self, model):
        """Verify embedding an over-length upstream sequence raises ValueError."""
        sequences = ["A" * 1004]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for upstream_species_lm must be <= 1003. Provided: 1004"
        )

    def test_embed_raises_value_error_for_downstream_short(self, model):
        """Verify embedding a too-short downstream sequence raises ValueError."""
        model.model_name = "downstream_species_lm"
        sequences = ["A" * 10]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for downstream_species_lm must be >  11. Provided: 10"
        )

    def test_embed_raises_value_error_for_downstream_long(self, model):
        """Verify embedding an over-length downstream sequence raises ValueError."""
        model.model_name = "downstream_species_lm"
        sequences = ["A" * 301]

        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences)
        assert (
            str(excinfo.value)
            == "Seq len for downstream_species_lm must be <= 300. Provided: 301"
        )

    def test_embed(self, model):
        """Verify embed returns tensors for both exact-length and padded inputs."""
        # Test with the correct sequence length
        sequences = ["ATTTG" * 200 + "ATG"][:1003]  # Adjusting to be exactly 1003 bp
        embedding = model.embed(sequences, mean_embedding=False)

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"

        # Test with sequence length that requires padding, and mean_embedding is True
        sequences = [
            "ATTTG" * 100 + "ATG"
        ]  # This sequence will be shorter than 1003 bp
        embedding = model.embed(
            sequences, mean_embedding=True
        )  # Adjust to mean_embedding=True

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"

    def test_embed_mean_embedding(self, model):
        """Verify embed returns a tensor when mean_embedding is enabled."""
        # Test the behavior of the embed method when mean_embedding is True
        sequences = ["ATTTG" * 200 + "ATG"]  # Example list of sequences
        embedding = model.embed(sequences, mean_embedding=True)

        # Validate the shape of the embedding and other necessary checks
        assert embedding is not None, "Embedding should not be None"
        assert isinstance(embedding, torch.Tensor), "Embedding should be a torch.Tensor"
        # Optionally, you can also check the values in the embedding Tensor

    def test_target_layer_as_int(self, model):
        """Verify an integer target layer yields a (1, 768) embedding."""
        model.target_layer = 1
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        embedding = model.embed(sequences, mean_embedding=True)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 768)

    def test_target_layer_as_single_element_tuple(self, model):
        """Verify a multi-layer tuple target yields a (1, 768) embedding."""
        model.target_layer = (8, 10)
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        embedding = model.embed(sequences, mean_embedding=True)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 768)

    def test_target_layer_as_single_element_tuple_error(self, model):
        """Verify an out-of-range target layer raises ValueError."""
        model.target_layer = (8, 14)
        sequences = ["ATTTG" * 200 + "ATG"]  # Adjust as per your needs
        with pytest.raises(ValueError) as excinfo:
            model.embed(sequences, mean_embedding=True)
        assert str(excinfo.value) == "Target layer 14 is out of range. Max layer is 13."


# ---------------------------------------------------------------------------
# Hermetic tests on a faked tokenizer and model (Phase 18).
# ---------------------------------------------------------------------------

HUB_DIR = "gagneurlab/SpeciesLM"
N_HIDDEN = 13


class _FakeSpeciesTokenizer:
    """One id per space-separated word between a CLS (0) and a SEP (1)."""

    def __init__(self) -> None:
        self.texts: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        """Record the text and return a batch of one."""
        self.texts.append((text, kwargs))
        words = text.split(" ")
        ids = [0] + [3 + i for i in range(len(words))] + [1]
        return {
            "input_ids": torch.tensor([ids]),
            "token_type_ids": torch.zeros((1, len(ids)), dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class _FakeHidden:
    """The attribute the wrapper reads from the model output."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeSpeciesLM:
    """Returns ``hidden_states[l][0, t] = [t, l]`` for 13 states."""

    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []
        self.grad_enabled: list[bool] = []
        self.eval_calls = 0

    def eval(self) -> "_FakeSpeciesLM":
        """Count the eval switch."""
        self.eval_calls += 1
        return self

    def __call__(self, **kwargs: torch.Tensor) -> _FakeHidden:
        """Record the inputs and whether autograd was on."""
        self.calls.append(kwargs)
        self.grad_enabled.append(torch.is_grad_enabled())
        length = kwargs["input_ids"].shape[-1]
        t = torch.arange(length, dtype=torch.float32)
        states = tuple(
            torch.stack([t, torch.full((length,), float(layer))], dim=-1).unsqueeze(0)
            for layer in range(N_HIDDEN)
        )
        return _FakeHidden(states)


class _HubRecorder:
    """Stands in for ``AutoTokenizer`` or ``AutoModelForMaskedLM``."""

    def __init__(self, tag: str, log: list[tuple[Any, ...]], obj: object) -> None:
        self.tag = tag
        self.log = log
        self.obj = obj

    def from_pretrained(self, *args: object, **kwargs: object) -> object:
        """Log the call and return the fake."""
        self.log.append((self.tag, args, kwargs))
        return self.obj


class _Faked:
    """The call log and the two fakes one construction was given."""

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.tokenizer = _FakeSpeciesTokenizer()
        self.model = _FakeSpeciesLM()


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Faked:
    """Replace the Hub loaders and the source location for every construction."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(fudt_module, "__file__", str(tmp_path / "fudt.py"))
    faked = _Faked()
    monkeypatch.setattr(
        fudt_module, "AutoTokenizer", _HubRecorder("tok", faked.log, faked.tokenizer)
    )
    monkeypatch.setattr(
        fudt_module,
        "AutoModelForMaskedLM",
        _HubRecorder("model", faked.log, faked.model),
    )
    return faked


def test_construction_downloads_the_revision_then_loads_it_in_eval_mode(
    fakes: _Faked, tmp_path: Path
) -> None:
    """A cold cache fetches both parts by ``revision`` into the cache, then loads.

    The model name is a git revision of the single Hub repo ``gagneurlab/SpeciesLM``;
    the cache is ``<source dir>/pretrained_LLM/fungal_up_down_transformer``; the loaded
    model is switched to eval once.
    """
    model = FungalUpDownTransformer(model_name="downstream_species_lm")
    cache = str(tmp_path / "pretrained_LLM" / "fungal_up_down_transformer")
    rev = {"revision": "downstream_species_lm"}
    assert fakes.log == [
        ("tok", (HUB_DIR,), {**rev, "cache_dir": cache}),
        ("model", (HUB_DIR,), {**rev, "cache_dir": cache}),
        ("tok", (HUB_DIR,), rev),
        ("model", (HUB_DIR,), rev),
    ]
    assert model.tokenizer is fakes.tokenizer
    assert model.model is fakes.model
    assert fakes.model.eval_calls == 1


def test_warm_cache_skips_the_download_and_prints_the_directory(
    fakes: _Faked, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``<cache>/gagneurlab/SpeciesLM/<revision>`` present only the loads run."""
    cached = (
        tmp_path
        / "pretrained_LLM"
        / "fungal_up_down_transformer"
        / HUB_DIR
        / "upstream_species_lm"
    )
    cached.mkdir(parents=True)
    FungalUpDownTransformer(model_name="upstream_species_lm")
    rev = {"revision": "upstream_species_lm"}
    assert fakes.log == [("tok", (HUB_DIR,), rev), ("model", (HUB_DIR,), rev)]
    assert capsys.readouterr().out == f"{cached} model already downloaded.\n"


def test_model_name_vocabulary_is_not_enforced(fakes: _Faked) -> None:
    """Finding: ``VALID_MODEL_NAMES`` is never checked; the prefix alone decides.

    The class docstring says the agnostic models "are not supported here", yet
    ``upstream_agnostic_lm`` constructs and takes the upstream limit 1003. The default
    ``model_name=""`` also constructs (loading revision "") and is refused only when a
    limit is needed, by ``max_sequence_size`` (fungal_up_down_transformer.py line 111)
    with the exact message below. Pinned until the constructor checks the vocabulary.
    """
    agnostic = FungalUpDownTransformer(model_name="upstream_agnostic_lm")
    assert agnostic.max_sequence_size == 1003
    unnamed = FungalUpDownTransformer()
    assert fakes.log[-1] == ("model", (HUB_DIR,), {"revision": ""})
    with pytest.raises(ValueError) as excinfo:
        unnamed.embed(["ACGTACGTACGT"])
    assert str(excinfo.value) == "Unknown model_name: "


def test_sequence_is_tokenized_as_species_then_stride_one_six_mers(
    fakes: _Faked,
) -> None:
    """``ACGTACGTACGT`` (12 bp) becomes the species token and 7 overlapping 6-mers.

    Six-mers at offsets 0 to 6: ACGTAC CGTACG GTACGT TACGTA ACGTAC CGTACG GTACGT. The
    default proxy species is ``candida_glabrata``; ``proxy_species`` replaces it. The
    species is one token either way, so both pool to [4.5, 10] (10 tokens).
    """
    model = FungalUpDownTransformer(model_name="downstream_species_lm")
    default = model.embed(["ACGTACGTACGT"])
    other = model.embed(["ACGTACGTACGT"], proxy_species="kluyveromyces_lactis")
    torch.testing.assert_close(default, torch.tensor([[4.5, 10.0]]))
    torch.testing.assert_close(other, torch.tensor([[4.5, 10.0]]))
    kmers = "ACGTAC CGTACG GTACGT TACGTA ACGTAC CGTACG GTACGT"
    assert fakes.tokenizer.texts == [
        (f"candida_glabrata {kmers}", {"return_tensors": "pt"}),
        (f"kluyveromyces_lactis {kmers}", {"return_tensors": "pt"}),
    ]


def test_downstream_length_window_is_eleven_to_three_hundred(fakes: _Faked) -> None:
    """Finding: 11 bp is accepted although the refusal text says "must be >  11".

    The check is ``sequence_length < 11`` (fungal_up_down_transformer.py line 189), so
    the window is 11 to 300 inclusive. 11 bp pools to [4, 10] (9 tokens, mean position
    4; layers 8 to 12) and 300 bp to [148.5, 10] (298 tokens, mean position 297 / 2).
    Pinned until the message and the check agree.
    """
    model = FungalUpDownTransformer(model_name="downstream_species_lm")
    torch.testing.assert_close(model.embed(["A" * 11]), torch.tensor([[4.0, 10.0]]))
    torch.testing.assert_close(model.embed(["A" * 300]), torch.tensor([[148.5, 10.0]]))
    with pytest.raises(ValueError) as short:
        model.embed(["A" * 10])
    assert str(short.value) == (
        "Seq len for downstream_species_lm must be >  11. Provided: 10"
    )
    with pytest.raises(ValueError) as long:
        model.embed(["A" * 301])
    assert str(long.value) == (
        "Seq len for downstream_species_lm must be <= 300. Provided: 301"
    )


def test_pad_sequence_inserts_zeros_after_the_first_two_tokens() -> None:
    """Four tokens pad to 1001: two kept, 997 zeros, then the other two.

    ``pad_length = 1001 - 4 = 997``, ``pad_start = 2``, ``pad_end = 999``; the same
    splice is applied to all three tensors.
    """
    data = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "token_type_ids": torch.tensor([[0, 0, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    out, pad_start, pad_end = FungalUpDownTransformer._pad_sequence(
        data, mean_embedding=True
    )
    zeros = torch.zeros(997, dtype=torch.long)
    assert (pad_start, pad_end) == (2, 999)
    assert torch.equal(
        out["input_ids"],
        torch.cat([torch.tensor([7, 8]), zeros, torch.tensor([9, 10])]).unsqueeze(0),
    )
    assert torch.equal(
        out["token_type_ids"],
        torch.cat([torch.tensor([0, 0]), zeros, torch.tensor([1, 1])]).unsqueeze(0),
    )
    assert torch.equal(
        out["attention_mask"],
        torch.cat([torch.tensor([1, 1]), zeros, torch.tensor([1, 1])]).unsqueeze(0),
    )


def test_short_upstream_is_padded_and_the_pad_rows_are_dropped_before_pooling(
    fakes: _Faked,
) -> None:
    """A 10 bp upstream sequence pools to [748.25, 10], not the all-row mean 500.

    Worked in the module docstring. The model sees 1001 ids with zeros at positions 2
    to 994 and a zero attention mask there, and runs with autograd off.
    """
    model = FungalUpDownTransformer(model_name="upstream_species_lm")
    out = model.embed(["ACGTACGTAC"])
    torch.testing.assert_close(out, torch.tensor([[748.25, 10.0]]))
    (seen,) = fakes.model.calls
    ids = seen["input_ids"][0]
    assert ids.shape == (1001,)
    assert ids[:2].tolist() == [0, 3]
    assert ids[995:].tolist() == [4, 5, 6, 7, 8, 1]
    assert int(ids[2:995].abs().sum()) == 0
    assert int(seen["attention_mask"].sum()) == 8
    assert fakes.model.grad_enabled == [False]


def test_short_upstream_without_mean_pooling_is_refused(fakes: _Faked) -> None:
    """Per-token output of a padded sequence trips the ``_pad_sequence`` assertion.

    The message (fungal_up_down_transformer.py line 131) is pinned verbatim, typo
    included ("meaning embedding").
    """
    model = FungalUpDownTransformer(model_name="upstream_species_lm")
    with pytest.raises(AssertionError) as excinfo:
        model.embed(["ACGTACGTAC"], mean_embedding=False)
    assert str(excinfo.value) == (
        "sequences must be 1003 bp if not using meaning embedding"
    )


def test_full_length_upstream_is_not_padded_and_one_base_more_is_refused(
    fakes: _Faked,
) -> None:
    """1003 bp gives 1001 tokens, no padding, and the per-token rows [t, 10].

    ``mean_embedding=False`` stacks each sequence's ``[tokens, dim]`` block, so the
    output is ``[1, 1001, 2]`` with row t equal to [t, 10] (layers 8 to 12). One
    base more, 1004, is refused with the exact message below.
    """
    model = FungalUpDownTransformer(model_name="upstream_species_lm")
    out = model.embed(["A" * 1003], mean_embedding=False)
    t = torch.arange(1001, dtype=torch.float32)
    expected = torch.stack([t, torch.full((1001,), 10.0)], dim=-1).unsqueeze(0)
    torch.testing.assert_close(out, expected)
    assert int(fakes.model.calls[0]["attention_mask"].sum()) == 1001
    with pytest.raises(ValueError) as excinfo:
        model.embed(["A" * 1004])
    assert str(excinfo.value) == (
        "Seq len for upstream_species_lm must be <= 1003. Provided: 1004"
    )


@pytest.mark.parametrize(
    ("target_layer", "layer_mean"),
    [(3, 3.0), ((8,), 10.0), ((2, 5), 3.5), ((0, 0), 0.0)],
)
def test_target_layer_selects_or_averages_hidden_states(
    fakes: _Faked, target_layer: int | tuple[int, ...], layer_mean: float
) -> None:
    """An int picks one state; ``(a,)`` averages a to the last; ``(a, b)`` a to b.

    Second coordinate: layer 3; layers 8 to 12 average 10; layers 2 to 5 average 3.5
    (b is inclusive); layer 0 alone. The first coordinate is the 12 bp mean 4.5.
    """
    model = FungalUpDownTransformer(
        model_name="downstream_species_lm", target_layer=target_layer
    )
    torch.testing.assert_close(
        model.embed(["ACGTACGTACGT"]), torch.tensor([[4.5, layer_mean]])
    )


def test_target_layer_upper_bound_is_off_by_one(fakes: _Faked) -> None:
    """Finding: ``(2, 13)`` passes the range check and silently averages 2 to 12.

    With 13 hidden states the largest index is 12, but the check is
    ``target_layer[1] > len(hidden_states)`` (fungal_up_down_transformer.py line 226),
    so 13 is accepted and the slice ``[2:14]`` stops at 12: mean layer 7. Only 14 is
    refused. Pinned until the check becomes ``>=``.
    """
    model = FungalUpDownTransformer(
        model_name="downstream_species_lm", target_layer=(2, 13)
    )
    torch.testing.assert_close(
        model.embed(["ACGTACGTACGT"]), torch.tensor([[4.5, 7.0]])
    )
    model.target_layer = (2, 14)
    with pytest.raises(ValueError) as excinfo:
        model.embed(["ACGTACGTACGT"])
    assert str(excinfo.value) == "Target layer 14 is out of range. Max layer is 13."


def test_batches_stack_per_sequence_in_input_order(fakes: _Faked) -> None:
    """Pooled rows come back ``[batch, dim]`` in order; per-token blocks stack.

    Pooled: 11 bp then 12 bp give [[4, 10], [4.5, 10]]. Per-token on two 12 bp
    sequences: ``[2, 10, 2]`` with row t equal to [t, 10] in both.
    """
    model = FungalUpDownTransformer(model_name="downstream_species_lm")
    pooled = model.embed(["A" * 11, "A" * 12])
    torch.testing.assert_close(pooled, torch.tensor([[4.0, 10.0], [4.5, 10.0]]))
    per_token = model.embed(["A" * 12, "C" * 12], mean_embedding=False)
    t = torch.arange(10, dtype=torch.float32)
    block = torch.stack([t, torch.full((10,), 10.0)], dim=-1)
    torch.testing.assert_close(per_token, torch.stack([block, block]))
