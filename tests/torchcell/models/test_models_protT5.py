# tests/torchcell/models/test_models_protT5.py
# [[tests.torchcell.models.test_models_protT5]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_models_protT5.py
"""The ProtT5 wrapper on a faked tokenizer and encoder, no weights loaded.

2026.10.06, Phase 21. The module-level names ``T5Tokenizer`` and ``T5EncoderModel``
are replaced by recorders whose ``from_pretrained`` returns a fake, the module's
``__file__`` is pointed into ``tmp_path`` (the download check builds its cache
directory next to the source file), CUDA is reported absent, and ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE`` are set as a second guard. The basename ``test_protT5.py`` is
taken by the dataset test, hence the ``test_models_`` prefix.

Fake tokenizer: the prepared string is split on spaces, each residue becomes token 5,
then EOS 1, right-padded with 0 to the longest row (``padding="longest"``); the mask
is 1 on non-pad positions. ``["AC", "ACGT"]`` is prepared to ``["A C", "A C G T"]``
and encodes to ``[[5, 5, 1, 0, 0], [5, 5, 5, 5, 1]]``, mask ``[[1, 1, 1, 0, 0], [1] * 5]``.

Fake encoder: ``last_hidden_state[b, t] = [t, 10 * b]`` times a unit weight that
requires grad (so the ``torch.no_grad`` around the forward is observable); it records
``half()`` / ``full()`` calls so the precision branch is observable.

Mean pooling (protT5.py:99 is ``embeddings.mean(dim=1)``, no mask): sequence 0
averages t over {0, ..., 4} = 10 / 5 = 2.0 (pads included); a masked mean would be
3 / 3 = 1.0. Sequence 1 averages 10 / 5 = 2.0, b-column 10.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import torchcell.models.protT5 as prot_module
from torchcell.models.protT5 import ProtT5

HUB_ID = "Rostlab/prot_t5_xl_uniref50"
CallLog = list[tuple[str, tuple[Any, ...], dict[str, Any]]]


class _FakeTokenizer:
    """Residue 5 per space-separated symbol, EOS 1, right-padded with 0."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, sequences: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Record the call and return padded ids plus the non-pad mask."""
        self.calls.append((list(sequences), kwargs))
        rows = [[5] * len(seq.split(" ")) + [1] for seq in sequences]
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class _FakeEncoder:
    """``last_hidden_state[b, t] = [t, 10 * b]``; records device and precision calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []
        self.events: list[str] = []
        # A unit weight that requires grad: outside ``torch.no_grad`` the output
        # would carry a graph, so the wrapper's ``no_grad`` is observable.
        self.scale = torch.ones((), requires_grad=True)

    def to(self, device: torch.device) -> "_FakeEncoder":
        """Record the device move."""
        self.events.append(f"to:{device}")
        return self

    def half(self) -> "_FakeEncoder":
        """Record a half-precision cast."""
        self.events.append("half")
        return self

    def full(self) -> "_FakeEncoder":
        """Record the call the CPU branch makes (a real ``nn.Module`` has none)."""
        self.events.append("full")
        return self

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        """Record the call and return the hidden state."""
        self.calls.append(kwargs)
        batch, length = kwargs["input_ids"].shape
        t = torch.arange(length, dtype=torch.float32).expand(batch, length)
        b = 10.0 * torch.arange(batch, dtype=torch.float32).unsqueeze(1).expand(
            batch, length
        )
        return SimpleNamespace(
            last_hidden_state=torch.stack([t, b], dim=-1) * self.scale
        )


class _Recorder:
    """Stands in for ``T5Tokenizer`` or ``T5EncoderModel``."""

    def __init__(self, tag: str, log: CallLog, obj: Any) -> None:
        self.tag = tag
        self.log = log
        self.obj = obj

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Log the call and hand back the fake."""
        self.log.append((self.tag, args, kwargs))
        return self.obj


@pytest.fixture
def faked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[CallLog, _FakeTokenizer, _FakeEncoder]:
    """Replace the Hub loaders and the source location; report CUDA absent."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(prot_module, "__file__", str(tmp_path / "protT5.py"))
    log: CallLog = []
    tok = _FakeTokenizer()
    model = _FakeEncoder()
    monkeypatch.setattr(prot_module, "T5Tokenizer", _Recorder("tok", log, tok))
    monkeypatch.setattr(prot_module, "T5EncoderModel", _Recorder("model", log, model))
    return log, tok, model


def test_construction_downloads_with_legacy_tokenizer_then_loads(
    faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cold cache: two ``cache_dir`` downloads, then the two loads, in that order.

    Pins: the Hub id ``Rostlab/<name>``; the download passes ``legacy=True`` to the
    tokenizer; the load passes ``do_lower_case=False`` (ProtT5 vocabularies are upper
    case); the cache directory ``<source dir>/pretrained_LLM/ProtT5`` on disk; stdout.
    """
    log, tok, model = faked
    wrapper = ProtT5("prot_t5_xl_uniref50")
    cache = str(tmp_path / "pretrained_LLM" / "ProtT5")
    assert wrapper.model_name == HUB_ID
    assert log == [
        ("tok", (HUB_ID,), {"cache_dir": cache, "legacy": True}),
        ("model", (HUB_ID,), {"cache_dir": cache}),
        ("tok", (HUB_ID,), {"do_lower_case": False}),
        ("model", (HUB_ID,), {}),
    ]
    assert os.path.isdir(cache)
    assert capsys.readouterr().out == (
        f"Downloading {HUB_ID} model to {cache}/{HUB_ID}...\nDownload finished.\n"
    )
    assert wrapper.tokenizer is tok
    assert wrapper.model is model
    assert wrapper.max_sequence_size == 40000


def test_warm_cache_skips_the_download(
    faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``<cache>/Rostlab/<name>`` present only the two plain loads run."""
    log, _, _ = faked
    (tmp_path / "pretrained_LLM" / "ProtT5" / HUB_ID).mkdir(parents=True)
    ProtT5("prot_t5_xl_uniref50")
    assert log == [
        ("tok", (HUB_ID,), {"do_lower_case": False}),
        ("model", (HUB_ID,), {}),
    ]
    assert capsys.readouterr().out == f"{HUB_ID} model already downloaded.\n"


def test_cpu_model_is_cast_to_half_precision(
    faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder],
) -> None:
    """Finding: on CPU the encoder is cast to fp16, the opposite of the comment.

    protT5.py:75 reads ``self.model.full() if self.device == "cpu" else
    self.model.half()`` under the comment "full-precision if on CPU". A
    ``torch.device`` never equals the string ``"cpu"`` (``torch.device("cpu") ==
    "cpu"`` is False, asserted below), so the CPU path always takes ``half()``. The
    other branch is unreachable, and would raise anyway: ``torch.nn.Module`` has no
    ``full`` method (also asserted). Pinned until the comparison is on
    ``self.device.type`` and the CPU branch calls ``float()``.
    """
    _, _, model = faked
    wrapper = ProtT5("prot_t5_xl_uniref50")
    assert wrapper.device == torch.device("cpu")
    assert (torch.device("cpu") == "cpu") is False  # type: ignore[comparison-overlap, unused-ignore]
    assert hasattr(torch.nn.Module, "full") is False
    assert model.events == ["to:cpu", "half"]


@pytest.mark.parametrize(
    ("raw", "prepared"),
    [
        ("MKUZOB", "M K X X X X"),
        ("ACDEFGHIKLMNPQRSTVWY", " ".join("ACDEFGHIKLMNPQRSTVWY")),
        ("X", "X"),
        ("", ""),
    ],
)
def test_prepare_sequence_maps_rare_residues_to_x_and_spaces_symbols(
    faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder], raw: str, prepared: str
) -> None:
    """U, Z, O and B become X (and only those); every symbol is space-separated.

    The 20 canonical residues pass through unchanged.
    """
    wrapper = ProtT5("prot_t5_xl_uniref50")
    assert wrapper._prepare_sequence(raw) == prepared


@pytest.fixture
def wrapper(faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder]) -> ProtT5:
    """A wrapper built on the fakes."""
    return ProtT5("prot_t5_xl_uniref50")


def test_embed_feeds_prepared_sequences_and_the_mask(
    wrapper: ProtT5, faked: tuple[CallLog, _FakeTokenizer, _FakeEncoder]
) -> None:
    """The tokenizer sees the prepared strings with the exact kwargs; the encoder the
    ids and mask of the module docstring, by keyword.

    No ``truncation`` / ``max_length`` is passed: ``max_sequence_size`` (40000) is
    never applied by ``embed``.
    """
    _, tok, model = faked
    wrapper.embed(["AC", "ACGU"])
    assert tok.calls == [
        (
            ["A C", "A C G X"],
            {"add_special_tokens": True, "padding": "longest", "return_tensors": "pt"},
        )
    ]
    (kwargs,) = model.calls
    assert set(kwargs) == {"input_ids", "attention_mask"}
    assert torch.equal(
        kwargs["input_ids"], torch.tensor([[5, 5, 1, 0, 0], [5, 5, 5, 5, 1]])
    )
    assert torch.equal(
        kwargs["attention_mask"], torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    )


def test_embed_per_token_returns_the_last_hidden_state(wrapper: ProtT5) -> None:
    """Without pooling the output is ``last_hidden_state`` unchanged, ``[2, 5, 2]``.

    The forward runs under ``torch.no_grad`` (protT5.py:90), so the output carries no
    graph although the fake's weight requires grad.
    """
    out = wrapper.embed(["AC", "ACGT"])
    t = torch.arange(5, dtype=torch.float32)
    expected = torch.stack(
        [
            torch.stack([t, torch.zeros(5)], dim=-1),
            torch.stack([t, torch.full((5,), 10.0)], dim=-1),
        ]
    )
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    assert out.requires_grad is False


def test_mean_embedding_averages_pad_positions(wrapper: ProtT5) -> None:
    """Finding: the mean includes padding, so a sequence's vector depends on its batch.

    protT5.py:99 is ``embeddings.mean(dim=1)`` with no attention mask. ``"AC"`` alone
    encodes to ``[5, 5, 1]`` and pools to t = 3 / 3 = 1.0; batched with ``"ACGT"`` it
    is padded to 5 positions and pools to 10 / 5 = 2.0. The ESM-2 wrapper masks its
    mean (esm2.py:102). ``ProtT5Dataset`` embeds one sequence per call
    (datasets/protT5.py:111), so stored embeddings have no padding, but the EOS
    position is still averaged in. Pinned until the mean is attention-masked (removes the batch dependence).
    """
    alone = wrapper.embed(["AC"], mean_embedding=True)
    batched = wrapper.embed(["AC", "ACGT"], mean_embedding=True)
    torch.testing.assert_close(alone, torch.tensor([[1.0, 0.0]]), rtol=0.0, atol=0.0)
    assert alone.requires_grad is False and batched.requires_grad is False
    torch.testing.assert_close(
        batched, torch.tensor([[2.0, 0.0], [2.0, 10.0]]), rtol=0.0, atol=0.0
    )
