# tests/torchcell/models/test_models_esm2.py
# [[tests.torchcell.models.test_models_esm2]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_models_esm2.py
"""The ESM-2 wrapper on a faked tokenizer and model, no weights loaded.

2026.10.06, Phase 21. The module-level names ``AutoTokenizer`` and
``AutoModelForMaskedLM`` are replaced by recorders whose ``from_pretrained`` returns a
fake, the module's ``__file__`` is pointed into ``tmp_path`` (the download check builds
its cache directory next to the source file), CUDA is reported absent, and
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard. Nothing reaches
the Hub. The basename ``test_esm2.py`` is taken by the dataset test, hence the
``test_models_`` prefix.

Fake tokenizer: each sequence encodes to ``[0] + [5] * len(seq) + [2]`` (CLS, residues,
EOS), right-padded with 1 to the longest row; the attention mask is 1 on every non-pad
position. ``["AC", "ACGT"]`` gives ids ``[[0, 5, 5, 2, 1, 1], [0, 5, 5, 5, 5, 2]]`` and
masks ``[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]``.

Fake model: ``hidden_states[-1][b, t] = [t, 10 * b]`` (shape ``[2, 6, 2]``), requires
grad (so ``detach`` is observable); ``hidden_states[0]`` is all -1 (so selecting the
wrong layer is observable).

Masked mean for ``["AC", "ACGT"]``: sequence 0 averages t over {0, 1, 2, 3}, 6 / 4 = 1.5,
b-column 0; sequence 1 averages t over {0, ..., 5}, 15 / 6 = 2.5, b-column 10. Ignoring
the mask would give 15 / 6 = 2.5 for sequence 0. The wrapper then adds a leading axis,
so the result is ``[[[1.5, 0], [2.5, 10]]]`` of shape ``[1, batch, dim]``;
``Esm2Dataset`` embeds one sequence per call and flattens that to ``[1, dim]`` with
``.reshape(1, -1)`` (datasets/esm2.py:166).

The truncation test uses the REAL ``transformers.EsmTokenizer`` built from a vocabulary
file written to ``tmp_path`` (the published ESM-2 33-token vocabulary), so it needs no
download.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub.file_download import repo_folder_name
from transformers import EsmTokenizer

import torchcell.models.esm2 as esm2_module
from torchcell.models.esm2 import Esm2

CHECKPOINT = "esm2_t6_8M_UR50D"
HUB_ID = f"facebook/{CHECKPOINT}"
CallLog = list[tuple[str, tuple[Any, ...], dict[str, Any]]]

# The ESM-2 vocabulary in token-id order (facebook/esm2_* vocab.txt).
ESM_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I",
    "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
]  # fmt: skip


class _FakeTokenizer:
    """CLS 0, residue 5, EOS 2, right-padded with 1 to the longest row."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def batch_encode_plus(
        self, sequences: list[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Record the call and return padded ids plus the non-pad mask."""
        self.calls.append((list(sequences), kwargs))
        rows = [[0] + [5] * len(seq) + [2] for seq in sequences]
        width = max(len(r) for r in rows)
        ids = [r + [1] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


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
    monkeypatch.setattr(esm2_module, "__file__", str(tmp_path / "esm2.py"))
    log: CallLog = []
    tok = _FakeTokenizer()
    model = _FakeMaskedLM()
    monkeypatch.setattr(esm2_module, "AutoTokenizer", _Recorder("tok", log, tok))
    monkeypatch.setattr(
        esm2_module, "AutoModelForMaskedLM", _Recorder("model", log, model)
    )
    return log, tok, model


def test_construction_downloads_into_cache_then_loads_by_hub_id(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cold cache fetches both parts with ``cache_dir``, then loads them without it.

    Pins: the Hub id is ``facebook/<checkpoint>``; the call order (download tokenizer,
    download model, load tokenizer, load model); the cache directory
    ``<source dir>/pretrained_LLM/Esm2`` created on disk; the exact stdout; the fakes
    stored on the instance; the model moved to CPU once (CUDA reported absent).
    """
    log, tok, model = faked
    wrapper = Esm2(CHECKPOINT)
    cache = str(tmp_path / "pretrained_LLM" / "Esm2")
    assert wrapper.model_name == HUB_ID
    assert log == [
        ("tok", (HUB_ID,), {"cache_dir": cache}),
        ("model", (HUB_ID,), {"cache_dir": cache}),
        ("tok", (HUB_ID,), {}),
        ("model", (HUB_ID,), {}),
    ]
    assert os.path.isdir(cache)
    assert capsys.readouterr().out == (
        f"Downloading {HUB_ID} model to {cache}/{HUB_ID}...\nDownload finished.\n"
    )
    assert wrapper.tokenizer is tok
    assert wrapper.model is model
    assert wrapper.device == torch.device("cpu")
    assert model.moved_to == [torch.device("cpu")]
    assert wrapper.max_sequence_size == 1022


def test_warm_cache_skips_the_download_and_says_so(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``<cache>/facebook/<checkpoint>`` present only the two plain loads run."""
    log, _, _ = faked
    (tmp_path / "pretrained_LLM" / "Esm2" / HUB_ID).mkdir(parents=True)
    Esm2(CHECKPOINT)
    assert log == [("tok", (HUB_ID,), {}), ("model", (HUB_ID,), {})]
    assert capsys.readouterr().out == f"{HUB_ID} model already downloaded.\n"


def test_download_check_never_matches_the_hub_cache_layout(
    faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM], tmp_path: Path
) -> None:
    """Finding: the "already downloaded" check looks for a directory the Hub never writes.

    ``from_pretrained(..., cache_dir=C)`` stores a repo under
    ``C/models--facebook--<checkpoint>`` (``huggingface_hub.repo_folder_name``), but the
    check at esm2.py:46-47 tests ``C/facebook/<checkpoint>``. With the Hub layout
    present (what a real first download leaves), a second construction still runs the
    two ``cache_dir`` downloads; and the loads at esm2.py:63-64 pass no ``cache_dir``,
    so they read the default HF cache, never the copy fetched into ``C``. Pinned until
    the check and the loads agree on one cache location.
    """
    log, _, _ = faked
    cache = tmp_path / "pretrained_LLM" / "Esm2"
    layout = repo_folder_name(repo_id=HUB_ID, repo_type="model")
    assert layout == "models--facebook--esm2_t6_8M_UR50D"
    (cache / layout).mkdir(parents=True)
    wrapper = Esm2(CHECKPOINT)
    assert wrapper.model_name == HUB_ID
    assert log == [
        ("tok", (HUB_ID,), {"cache_dir": str(cache)}),
        ("model", (HUB_ID,), {"cache_dir": str(cache)}),
        ("tok", (HUB_ID,), {}),
        ("model", (HUB_ID,), {}),
    ]


@pytest.fixture
def wrapper(faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]) -> Esm2:
    """A wrapper built on the fakes (cold-cache path, pinned above)."""
    return Esm2(CHECKPOINT)


def test_embed_tokenizes_with_truncation_to_max_sequence_size(
    wrapper: Esm2, faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]
) -> None:
    """The tokenizer gets the exact kwargs; the model gets the ids and the mask.

    ``["AC", "ACGT"]`` encodes to ``[[0, 5, 5, 2, 1, 1], [0, 5, 5, 5, 5, 2]]`` with mask
    ``[[1, 1, 1, 1, 0, 0], [1] * 6]`` (module docstring); ``output_hidden_states`` is
    requested because the wrapper reads ``hidden_states[-1]``.
    """
    _, tok, model = faked
    wrapper.embed(["AC", "ACGT"])
    assert tok.calls == [
        (
            ["AC", "ACGT"],
            {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": 1022,
            },
        )
    ]
    ((ids, kwargs),) = model.calls
    assert torch.equal(ids, torch.tensor([[0, 5, 5, 2, 1, 1], [0, 5, 5, 5, 5, 2]]))
    assert set(kwargs) == {"attention_mask", "output_hidden_states"}
    assert torch.equal(
        kwargs["attention_mask"], torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    )
    assert kwargs["output_hidden_states"] is True


def test_embed_per_token_returns_last_hidden_state_detached(wrapper: Esm2) -> None:
    """Without pooling the output is the LAST hidden state, pads included, detached."""
    out = wrapper.embed(["AC", "ACGT"])
    t = torch.arange(6, dtype=torch.float32)
    expected = torch.stack(
        [
            torch.stack([t, torch.zeros(6)], dim=-1),
            torch.stack([t, torch.full((6,), 10.0)], dim=-1),
        ]
    )
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    assert out.requires_grad is False


def test_embed_mean_pools_over_unmasked_tokens_with_a_leading_axis(
    wrapper: Esm2,
) -> None:
    """The masked mean is ``[[[1.5, 0], [2.5, 10]]]``, shape ``[1, 2, 2]``.

    Arithmetic in the module docstring; an unmasked mean would make the first row 2.5.
    """
    out = wrapper.embed(["AC", "ACGT"], mean_embedding=True)
    assert out.shape == (1, 2, 2)
    torch.testing.assert_close(
        out, torch.tensor([[[1.5, 0.0], [2.5, 10.0]]]), rtol=0.0, atol=1e-6
    )


def test_embed_wraps_a_bare_string_into_a_batch_of_one(
    wrapper: Esm2, faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]
) -> None:
    """``"ACG"`` is sent as ``["ACG"]``: ids ``[0, 5, 5, 5, 2]``, mean t = 10 / 5 = 2."""
    _, tok, _ = faked
    out = wrapper.embed("ACG", mean_embedding=True)
    assert tok.calls[0][0] == ["ACG"]
    torch.testing.assert_close(out, torch.tensor([[[2.0, 0.0]]]), rtol=0.0, atol=0.0)


def test_real_esm_tokenizer_truncation_keeps_1020_residues(
    wrapper: Esm2, tmp_path: Path, faked: tuple[CallLog, _FakeTokenizer, _FakeMaskedLM]
) -> None:
    """Finding: ``max_sequence_size`` 1022 is spent on TOKENS, so 1020 residues survive.

    ESM-2 has 1024 positions: 1022 residues plus CLS and EOS. esm2.py:88 passes
    ``max_length=self.max_sequence_size`` (1022) to ``batch_encode_plus``, whose
    ``max_length`` counts the special tokens, so a 1024-residue protein is cut to
    1022 tokens = CLS + 1020 residues + EOS, two residues fewer than the model
    accepts. Verified with the real ``EsmTokenizer`` on the published vocabulary
    (``A`` is id 5, CLS 0, EOS 2). Pinned until ``max_length`` is
    ``max_sequence_size + 2`` or the property is redefined as a token budget.
    """
    _, _, model = faked
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(ESM_VOCAB) + "\n")
    wrapper.tokenizer = EsmTokenizer(vocab_file=str(vocab))
    wrapper.embed(["A" * 1024])
    ((ids, kwargs),) = model.calls
    assert ids.shape == (1, 1022)
    assert ids[0, 0].item() == 0 and ids[0, -1].item() == 2
    assert torch.equal(ids[0, 1:-1], torch.full((1020,), 5))
    assert torch.equal(kwargs["attention_mask"], torch.ones(1, 1022, dtype=torch.long))
