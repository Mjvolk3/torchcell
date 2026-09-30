# tests/torchcell/models/test_llm.py
# [[tests.torchcell.models.test_llm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_llm.py
"""The language-model interfaces in ``torchcell.models.llm`` with no weights loaded.

2026.09.30, Phase 17. The module holds only two abstract bases,
``NucleotideModel`` and ``PeptideModel``, and the attrs container ``pretrained_LLM``;
it has no tokenization helpers, window arithmetic or output selection of its own (those
live in the concrete subclasses under ``torchcell.models``), and nothing in it
downloads weights. What a fake can reach is therefore the whole module:

* ``__init__`` sets ``tokenizer`` and ``model`` to None and THEN calls
  ``load_model(model_name)`` exactly once, so a subclass's loader sees the reset
  attributes and its assignments survive construction.
* ``max_sequence_size`` returns ``_max_sequence_size`` and refuses None with
  ``ValueError("Max size has not been set for this model.")``.
* Instantiating a subclass that leaves any of the three abstract methods
  unimplemented raises TypeError naming the missing ones, in sorted order.
* The abstract bodies are ``pass``, so calling them through the base returns None
  (a Finding: ``super().embed(...)`` gives None, not NotImplementedError).
* ``pretrained_LLM`` is an attrs class: positional order (tokenizer, model), value
  equality over both fields, and no validation of the declared ``AutoTokenizer`` /
  ``AutoModelForMaskedLM`` types.

The concrete subclasses are defined in the test module; they are model interfaces, not
schema classes. The two bases are identical in behavior, so each test runs on both.
"""

import re
from typing import Any

import attrs
import pytest
import torch

from torchcell.models.llm import NucleotideModel, PeptideModel, pretrained_LLM


class _FakeNucleotide(NucleotideModel):
    """Records what ``load_model`` saw and installs tiny stand-ins.

    Sets ``tokenizer`` and ``model`` to "stale" before calling the base constructor, so
    the recorded loader call shows whether the base reset them first.
    """

    def __init__(self, model_name: str, max_size: int | None) -> None:
        self._max_sequence_size = max_size
        self.seen: list[tuple[str, Any, Any]] = []
        self.tokenizer = "stale"
        self.model = "stale"
        super().__init__(model_name)

    @staticmethod
    def _check_and_download_model() -> None:
        raise AssertionError("no download in tests")

    def load_model(self, model_name: str) -> None:
        self.seen.append((model_name, self.tokenizer, self.model))
        self.tokenizer = f"tokenizer:{model_name}"
        self.model = torch.nn.Identity()

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        return torch.tensor([[float(len(s))] for s in sequences])


class _FakePeptide(PeptideModel):
    """Same recorder on the peptide base."""

    def __init__(self, model_name: str, max_size: int | None) -> None:
        self._max_sequence_size = max_size
        self.seen: list[tuple[str, Any, Any]] = []
        self.tokenizer = "stale"
        self.model = "stale"
        super().__init__(model_name)

    @staticmethod
    def _check_and_download_model(model_name: str) -> None:
        raise AssertionError("no download in tests")

    def load_model(self, model_name: str) -> None:
        self.seen.append((model_name, self.tokenizer, self.model))
        self.tokenizer = f"tokenizer:{model_name}"
        self.model = torch.nn.Identity()

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        return torch.tensor([[float(len(s))] for s in sequences])


class _NucleotideWithoutEmbed(NucleotideModel):
    @staticmethod
    def _check_and_download_model() -> None:
        return None

    def load_model(self, model_name: str) -> None:
        return None


class _PeptideWithOnlyEmbed(PeptideModel):
    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        return torch.zeros(0)


FAKES = [_FakeNucleotide, _FakePeptide]


@pytest.mark.parametrize("fake", FAKES)
def test_init_resets_both_handles_then_loads_the_named_model_once(fake: Any) -> None:
    """The fake sets tokenizer and model to "stale" before the base constructor; the
    loader sees ("nt-tiny", None, None), so the reset precedes the one load, and the
    loader's stand-ins are what the instance keeps.
    """
    wrapper = fake("nt-tiny", max_size=12)
    assert wrapper.seen == [("nt-tiny", None, None)]
    assert wrapper.tokenizer == "tokenizer:nt-tiny"
    assert type(wrapper.model) is torch.nn.Identity


@pytest.mark.parametrize("fake", FAKES)
def test_max_sequence_size_returns_the_set_value_and_refuses_none(fake: Any) -> None:
    """A set size is returned as is (including 0, which is not None); None raises the
    exact message.
    """
    assert fake("m", max_size=512).max_sequence_size == 512
    assert fake("m", max_size=0).max_sequence_size == 0
    with pytest.raises(
        ValueError, match=r"^Max size has not been set for this model\.$"
    ):
        fake("m", max_size=None).max_sequence_size  # noqa: B018


@pytest.mark.parametrize(
    ("cls", "missing"),
    [
        (NucleotideModel, "methods '_check_and_download_model', 'embed', 'load_model'"),
        (PeptideModel, "methods '_check_and_download_model', 'embed', 'load_model'"),
        (_NucleotideWithoutEmbed, "method 'embed'"),
        (_PeptideWithOnlyEmbed, "methods '_check_and_download_model', 'load_model'"),
    ],
)
def test_a_class_missing_abstract_methods_cannot_be_instantiated(
    cls: Any, missing: str
) -> None:
    """ABC refuses construction with the exact message, listing the unimplemented methods
    in sorted order (the Python 3.12+ wording).
    """
    message = (
        f"Can't instantiate abstract class {cls.__name__} without an implementation "
        f"for abstract {missing}"
    )
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        cls("m")


def test_pretrained_llm_is_a_positional_attrs_pair_with_value_equality() -> None:
    """Fields in order (tokenizer, model); equality compares both; the declared types are
    annotations only, so a stand-in tokenizer and module are accepted unconverted.
    """
    tokenizer: Any = "tok"
    other: Any = "other"
    model: Any = torch.nn.Identity()
    pair = pretrained_LLM(tokenizer, model)
    assert [f.name for f in attrs.fields(pretrained_LLM)] == ["tokenizer", "model"]
    assert (pair.tokenizer, pair.model) == (tokenizer, model)
    assert pair == pretrained_LLM(tokenizer=tokenizer, model=model)
    assert pair != pretrained_LLM(other, model)
    assert attrs.asdict(pair, recurse=False) == {"tokenizer": "tok", "model": model}


@pytest.mark.parametrize(
    ("base", "fake"), [(NucleotideModel, _FakeNucleotide), (PeptideModel, _FakePeptide)]
)
def test_the_abstract_bodies_return_none_when_a_subclass_calls_super(
    base: Any, fake: Any
) -> None:
    """Finding: the three abstract methods have ``pass`` bodies (llm.py:42, 47, 52 and
    82, 87, 92) rather than raising NotImplementedError, so a subclass that delegates
    with ``super().embed(...)`` silently receives None instead of an error or a
    tensor. Pinned until the bodies raise.
    """
    wrapper = fake("m", max_size=1)
    download_args: list[str] = [] if base is NucleotideModel else ["m"]
    assert base.embed(wrapper, ["ACGT"]) is None
    assert base.load_model(wrapper, "other") is None
    assert base._check_and_download_model(*download_args) is None
    assert wrapper.tokenizer == "tokenizer:m"
