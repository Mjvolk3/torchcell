# tests/torchcell/models/test_nucleotide_transformer.py
# [[tests.torchcell.models.test_nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_nucleotide_transformer.py
"""The Nucleotide Transformer wrapper on a faked tokenizer and model, no weights loaded.

2026.09.30, Phase 18. The module-level names ``AutoTokenizer`` and
``AutoModelForMaskedLM`` are replaced by recorders whose ``from_pretrained`` returns a
fake, the module's ``__file__`` is pointed into ``tmp_path`` (the download check builds
its cache directory next to the source file), CUDA is reported absent, and
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard. Nothing reaches
the Hub.

Fake tokenizer: ``model_max_length = 6``, ``pad_token_id = 1``; each sequence encodes
to ``[2] + [5] * len(seq)`` right-padded with 1 to length 6, so ``["AC", "ACGT"]``
gives ``[[2, 5, 5, 1, 1, 1], [2, 5, 5, 5, 5, 1]]`` and the attention mask
``ids != 1`` keeps positions 0 to 2 and 0 to 4.

Fake model: the last hidden state is ``h[b, t] = [t, 10 * b]`` (shape ``[2, 6, 2]``)
and requires grad, so the wrapper's ``detach`` is observable.

Worked values for ``mean_embedding=True``: sequence 0 averages t over {0, 1, 2}, which
is 3 / 3 = 1, with the b-column 0; sequence 1 averages t over {0, 1, 2, 3, 4}, which is
10 / 5 = 2, with the b-column 10. The pad positions (t = 3, 4, 5 for sequence 0) would
move the first mean to 15 / 6 = 2.5 if the mask were ignored. The wrapper then adds a
leading axis, so the result is ``[[[1, 0], [2, 10]]]`` of shape ``[1, 2, 2]`` (a
Finding: the mean path does not return ``[batch, dim]``).
"""

import os
from pathlib import Path
from typing import Any

import pytest
import torch

import torchcell.models.nucleotide_transformer as nt_module
from torchcell.models.nucleotide_transformer import NucleotideTransformer

HUB_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
CallLog = list[tuple[str, tuple[Any, ...], dict[str, Any]]]


class _FakeTokenizer:
    """Encodes a base as token 5 after a CLS token 2, right-pads with token 1."""

    model_max_length = 6
    pad_token_id = 1

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def batch_encode_plus(
        self, sequences: list[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Record the call and return right-padded ids."""
        self.calls.append((list(sequences), kwargs))
        rows = []
        for seq in sequences:
            ids = [2] + [5] * len(seq)
            rows.append(ids + [1] * (kwargs["max_length"] - len(ids)))
        return {"input_ids": torch.tensor(rows)}


class _FakeMaskedLM:
    """Returns ``hidden_states[-1][b, t] = [t, 10 * b]`` and records its inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, dict[str, Any]]] = []
        self.moved_to: list[torch.device] = []

    def to(self, device: torch.device) -> "_FakeMaskedLM":
        """Record the device move."""
        self.moved_to.append(device)
        return self

    def __call__(self, tokens_ids: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return two hidden-state tensors."""
        self.calls.append((tokens_ids, kwargs))
        batch, length = tokens_ids.shape
        t = torch.arange(length, dtype=torch.float32).expand(batch, length)
        b = 10.0 * torch.arange(batch, dtype=torch.float32).unsqueeze(1).expand(
            batch, length
        )
        last = torch.stack([t, b], dim=-1).requires_grad_(True)
        return {"hidden_states": (torch.full_like(last, -1.0), last)}


class _Recorder:
    """Stands in for ``AutoTokenizer`` or ``AutoModelForMaskedLM``."""

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
) -> tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]:
    """Replace the Hub loaders and the source location; report CUDA absent."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nt_module, "__file__", str(tmp_path / "nt.py"))
    log: CallLog = []
    tok = _FakeTokenizer()
    model = _FakeMaskedLM()
    monkeypatch.setattr(nt_module, "AutoTokenizer", _Recorder("tok", log, tok))
    monkeypatch.setattr(
        nt_module, "AutoModelForMaskedLM", _Recorder("model", log, model)
    )
    return log, tok, model


def test_construction_downloads_into_cache_then_loads_by_hub_id(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM], tmp_path: Path
) -> None:
    """A cold cache fetches both parts with ``cache_dir``, then loads them without it.

    Pins the call order (download tokenizer, download model, load tokenizer, load
    model), the cache directory ``<source dir>/pretrained_LLM/nucleotide_transformer``
    created on disk, the loaded fakes stored on the instance, and the model moved to
    the selected device once.
    """
    log, tok, model = faked
    wrapper = NucleotideTransformer()
    cache = str(tmp_path / "pretrained_LLM" / "nucleotide_transformer")
    assert log == [
        ("tok", (HUB_ID,), {"cache_dir": cache}),
        ("model", (HUB_ID,), {"cache_dir": cache}),
        ("tok", (HUB_ID,), {}),
        ("model", (HUB_ID,), {}),
    ]
    assert os.path.isdir(cache)
    assert wrapper.tokenizer is tok
    assert wrapper.model is model
    assert model.moved_to == [torch.device("cpu")]


def test_warm_cache_skips_the_download_and_says_so(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``<cache>/<hub id>`` present only the two plain loads run.

    The check is on the directory named by the Hub id itself (not the HF cache layout
    ``models--org--name``), and the only stdout line is the exact notice.
    """
    log, _, _ = faked
    (tmp_path / "pretrained_LLM" / "nucleotide_transformer" / HUB_ID).mkdir(
        parents=True
    )
    NucleotideTransformer()
    assert log == [("tok", (HUB_ID,), {}), ("model", (HUB_ID,), {})]
    assert capsys.readouterr().out == f"{HUB_ID} model already downloaded.\n"


def test_load_model_ignores_its_model_name_argument(  # test-quality: allow the contract is the loader call log the fakes record
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM], tmp_path: Path
) -> None:
    """Finding: ``load_model(model_name=...)`` always loads the module constant.

    ``load_model`` (nucleotide_transformer.py lines 55 to 65) passes ``MODEL_NAME`` to
    both ``from_pretrained`` calls, so the argument is dead. Pinned until the loader
    either honors or drops it. The reload also moves the model to the device again.
    """
    log, _, model = faked
    (tmp_path / "pretrained_LLM" / "nucleotide_transformer" / HUB_ID).mkdir(
        parents=True
    )
    wrapper = NucleotideTransformer()
    log.clear()
    wrapper.load_model(model_name="some/other-model")
    assert log == [("tok", (HUB_ID,), {}), ("model", (HUB_ID,), {})]
    assert model.moved_to == [torch.device("cpu"), torch.device("cpu")]


@pytest.fixture
def wrapper(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM],
) -> NucleotideTransformer:
    """A wrapper built on the fakes (cold-cache path, already pinned above)."""
    return NucleotideTransformer()


def test_embed_tokenizes_to_model_max_length_and_masks_pads(
    wrapper: NucleotideTransformer, faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]
) -> None:
    """The tokenizer and model receive exactly the padded ids and the pad mask.

    ``["AC", "ACGT"]`` encodes to ``[[2, 5, 5, 1, 1, 1], [2, 5, 5, 5, 5, 1]]``; the
    mask is ``ids != pad_token_id`` and is passed as both attention masks.
    """
    _, tok, model = faked
    wrapper.embed(["AC", "ACGT"])
    assert tok.calls == [
        (
            ["AC", "ACGT"],
            {"return_tensors": "pt", "padding": "max_length", "max_length": 6},
        )
    ]
    ((ids, kwargs),) = model.calls
    torch.testing.assert_close(
        ids, torch.tensor([[2, 5, 5, 1, 1, 1], [2, 5, 5, 5, 5, 1]])
    )
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, True, False]]
    )
    assert torch.equal(kwargs["attention_mask"], mask)
    assert torch.equal(kwargs["encoder_attention_mask"], mask)
    assert kwargs["output_hidden_states"] is True


def test_embed_per_token_returns_last_hidden_state_detached(
    wrapper: NucleotideTransformer,
) -> None:
    """Without pooling the output is the LAST hidden state, pads included, detached.

    The fake's first hidden state is all -1, so selecting index 0 would fail; the
    fake's last state requires grad, so a missing ``detach`` would fail.
    """
    out = wrapper.embed(["AC", "ACGT"])
    t = torch.arange(6, dtype=torch.float32)
    expected = torch.stack(
        [
            torch.stack([t, torch.zeros(6)], dim=-1),
            torch.stack([t, torch.full((6,), 10.0)], dim=-1),
        ]
    )
    torch.testing.assert_close(out, expected)
    assert out.requires_grad is False


def test_embed_mean_pools_over_unmasked_tokens_with_a_leading_axis(
    wrapper: NucleotideTransformer,
) -> None:
    """Finding: the masked mean comes back as ``[1, batch, dim]``, not ``[batch, dim]``.

    Worked in the module docstring: [[[1, 0], [2, 10]]]. The ``unsqueeze(0)`` at
    nucleotide_transformer.py line 102 adds the leading axis. Pinned until the pooled
    shape is decided.
    """
    out = wrapper.embed(["AC", "ACGT"], mean_embedding=True)
    torch.testing.assert_close(out, torch.tensor([[[1.0, 0.0], [2.0, 10.0]]]))


def test_embed_wraps_a_bare_string_into_a_batch_of_one(
    wrapper: NucleotideTransformer, faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]
) -> None:
    """A single string is encoded as ``["ACG"]``: mean over t in {0..3} is 1.5.

    ``"ACG"`` encodes to ``[2, 5, 5, 5, 1, 1]``, four unmasked positions, so the mean
    of t is 6 / 4 = 1.5 and the b-column is 0; the output is ``[[[1.5, 0.0]]]``.
    """
    _, tok, _ = faked
    out = wrapper.embed("ACG", mean_embedding=True)
    assert tok.calls[0][0] == ["ACG"]
    torch.testing.assert_close(out, torch.tensor([[[1.5, 0.0]]]))
