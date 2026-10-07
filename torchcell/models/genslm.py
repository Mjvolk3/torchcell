# torchcell/models/genslm.py
# [[torchcell.models.genslm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/genslm.py
# Test file: tests/torchcell/models/test_models_genslm.py

"""GenSLM codon language model (Zvyagin et al. 2023) wrapper that embeds coding
sequences.

GenSLMs are decoder-only GPT-NeoX models pretrained on ">110 million unique
prokaryotic gene sequences from BV-BRC", with inputs "encoded at the codon level
(every three nucleotide represents a codon; hence the 20 natural amino acid language
is described by 64 codons)" (``zvyaginGenSLMsGenomescaleLanguage2023``, Sec. 5.1.2
and Fig. 1). The model sees a coding sequence only: no promoter, UTR or intergenic
sequence, and a context of 2,048 codons (6,144 nt).

The tokenizer and architecture files are vendored under ``genslm_assets/`` from the
MIT-licensed ``ramanathanlab/genslm`` repository (``provenance.json`` pins the commit
and every sha256), so the ``genslm`` package itself, whose pins (``pytorch-lightning
1.6.5``, a ``transformers`` fork, ``pydantic 1``) conflict with this environment, is
not a dependency. The checkpoints are released only through the authors' Globus
endpoint; ``weights_dir`` must hold the checkpoint named in :data:`GENSLM_MODELS`
beside a ``manifest.json`` recording how it was retrieved and its sha256, which is
verified before the weights are loaded.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

from torchcell.literature.manifest import RetrievalRecord, sha256_file
from torchcell.models.llm import NucleotideModel

ASSETS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "genslm_assets")
TOKENIZER_ASSET = "codon_wordlevel_69vocab.json"
#: Context length of every foundation model, in codons (Zvyagin 2023 Table 1, MSL).
GENSLM_SEQ_LENGTH = 2048
#: Where the checkpoints live: ``$DATA_ROOT/models/genslm/``.
WEIGHTS_SUBDIR = osp.join("models", "genslm")
WEIGHTS_MANIFEST = "manifest.json"
#: The Globus endpoint the README names as the only release channel.
GLOBUS_ENDPOINT_ID = "25918ad0-2a4e-4f37-bcfc-8183b19c3150"

#: Buffers a checkpoint may legitimately omit: rotary tables and the causal mask are
#: recomputed from the config. A missing *parameter* is never accepted.
_DERIVED_BUFFER_SUFFIXES = (
    "rotary_emb.inv_freq",
    "attention.bias",
    "attention.masked_bias",
)


class GenSLMModelSpec(BaseModel):
    """One released foundation model: its vendored config and its checkpoint name."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    config_asset: str = Field(
        description="File under genslm_assets/ (an absolute path is used as given)."
    )
    weights_file: str = Field(
        description="Checkpoint file name exactly as released on the Globus endpoint."
    )
    hidden_size: int


#: The foundation ("patric") models, as registered in ``genslm.inference.GenSLM.MODELS``.
GENSLM_MODELS: dict[str, GenSLMModelSpec] = {
    spec.model_id: spec
    for spec in (
        GenSLMModelSpec(
            model_id="genslm_25M_patric",
            config_asset="neox_25M.json",
            weights_file="patric_25m_epoch01-val_loss_0.57_bias_removed.pt",
            hidden_size=512,
        ),
        GenSLMModelSpec(
            model_id="genslm_250M_patric",
            config_asset="neox_250M.json",
            weights_file="patric_250m_epoch00_val_loss_0.48_attention_removed.pt",
            hidden_size=1840,
        ),
        GenSLMModelSpec(
            model_id="genslm_2.5B_patric",
            config_asset="neox_2.5B.json",
            weights_file="patric_2.5b_epoch00_val_los_0.29_bias_removed.pt",
            hidden_size=3840,
        ),
        GenSLMModelSpec(
            model_id="genslm_25B_patric",
            config_asset="neox_25B.json",
            weights_file="model-epoch00-val_loss0.70-v2.pt",
            hidden_size=8196,
        ),
    )
}


class GenSLMWeightsManifest(BaseModel):
    """``manifest.json`` in the weights directory: one retrieval record per checkpoint,
    keyed by file name. Written by the fetch script, read before every load.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    globus_endpoint_id: str = GLOBUS_ENDPOINT_ID
    files: dict[str, RetrievalRecord] = Field(default_factory=dict)


def default_weights_dir() -> str:
    """``$DATA_ROOT/models/genslm``; ``DATA_ROOT`` must be set."""
    return osp.join(os.environ["DATA_ROOT"], WEIGHTS_SUBDIR)


def load_weights_manifest(weights_dir: str | Path) -> GenSLMWeightsManifest:
    """Validate and return the weights manifest, or raise with the fetch recipe."""
    path = Path(weights_dir) / WEIGHTS_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. GenSLM checkpoints are released only through Globus "
            f"endpoint {GLOBUS_ENDPOINT_ID}; fetch them with "
            "scripts/genslm_fetch_weights.py, which writes this manifest."
        )
    return GenSLMWeightsManifest.model_validate_json(path.read_text())


def codon_tokens(sequence: str) -> str:
    """Space-joined codons of an in-frame coding sequence, as the tokenizer expects.

    Mirrors ``genslm.dataset.SequenceDataset.group_by_kmer`` (non-overlapping
    triplets from position 0, upper-cased) but refuses what that function would pass
    through silently: a length that is not a multiple of 3 (an out-of-frame tail
    codon), and any character outside ``ACGT`` (the word-level vocabulary maps such
    a codon to ``[UNK]``, which is not a measurement of the gene).
    """
    seq = sequence.upper()
    if len(seq) % 3 != 0:
        raise ValueError(
            f"coding sequence length {len(seq)} is not a multiple of 3; GenSLM "
            "tokenizes in-frame codons"
        )
    bad = sorted(set(seq) - set("ACGT"))
    if bad:
        raise ValueError(
            f"coding sequence contains non-ACGT characters {bad}; the codon "
            "vocabulary has no token for them"
        )
    return " ".join(seq[i : i + 3] for i in range(0, len(seq), 3))


class GenSLM(NucleotideModel):
    """GenSLM foundation model that embeds coding sequences codon by codon.

    ``embed`` returns the last hidden layer. With ``mean_embedding=True`` it is the
    mean over the real codon positions only (``[PAD]`` masked out), which is the
    README recipe (``hidden_states[-1]`` averaged over the sequence axis) made exact:
    the README averages over the padded length, so a short gene's vector would be
    diluted by pad positions in proportion to the batch's padding.
    """

    VALID_MODEL_NAMES = list(GENSLM_MODELS)

    def __init__(self, model_name: str, weights_dir: str | None = None) -> None:
        """Resolve the spec, verify the checkpoint's sha256, and load the model.

        Args:
            model_name: One of :data:`GENSLM_MODELS`, e.g. ``genslm_25M_patric``.
            weights_dir: Directory holding the checkpoint and ``manifest.json``;
                defaults to ``$DATA_ROOT/models/genslm``.
        """
        if model_name not in GENSLM_MODELS:
            raise ValueError(
                f"Invalid model_name {model_name!r}; valid: {self.VALID_MODEL_NAMES}"
            )
        self.spec = GENSLM_MODELS[model_name]
        self.weights_dir = weights_dir or default_weights_dir()
        self._max_sequence_size = GENSLM_SEQ_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_name)

    @staticmethod
    def _check_and_download_model() -> None:
        """Checkpoints are not downloadable from code (Globus login); see the manifest."""
        return None

    @property
    def max_sequence_size(self) -> int:
        """Context length in codons."""
        return GENSLM_SEQ_LENGTH

    @property
    def hidden_size(self) -> int:
        """Width of the returned embedding."""
        return self.spec.hidden_size

    @property
    def weights_path(self) -> str:
        """Absolute path of this model's checkpoint."""
        return osp.join(self.weights_dir, self.spec.weights_file)

    def verify_weights(self) -> str:
        """Return the checkpoint path after checking its sha256 against the manifest."""
        manifest = load_weights_manifest(self.weights_dir)
        if self.spec.weights_file not in manifest.files:
            raise KeyError(
                f"{self.spec.weights_file} has no retrieval record in "
                f"{osp.join(self.weights_dir, WEIGHTS_MANIFEST)}"
            )
        path = Path(self.weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is in the manifest but not on disk")
        pinned = manifest.files[self.spec.weights_file].sha256
        got = sha256_file(path)
        if got != pinned:
            raise ValueError(
                f"{path}: sha256 {got} on disk, manifest pins {pinned}; the checkpoint "
                "is not the one that was retrieved"
            )
        return str(path)

    @staticmethod
    def build_tokenizer() -> PreTrainedTokenizerFast:
        """The 69-token codon vocabulary (64 codons + 5 specials), ``[PAD]`` set."""
        tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # transformers tokenizer __init__ is untyped
            tokenizer_object=Tokenizer.from_file(osp.join(ASSETS_DIR, TOKENIZER_ASSET))
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def build_config(self) -> GPTNeoXConfig:
        """The GPT-NeoX config of this model (vendored, or an absolute path for tests)."""
        asset = self.spec.config_asset
        path = asset if osp.isabs(asset) else osp.join(ASSETS_DIR, asset)
        config = GPTNeoXConfig.from_json_file(path)
        assert isinstance(config, GPTNeoXConfig)
        return config

    def load_state_dict_strict(self, model: GPTNeoXForCausalLM, path: str) -> None:
        """Load a Lightning checkpoint's ``state_dict``; only derived buffers may be
        absent, and nothing unexpected may be present.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict: dict[str, torch.Tensor] = checkpoint["state_dict"]
        result = model.load_state_dict(state_dict, strict=False)
        missing = [
            k for k in result.missing_keys if not k.endswith(_DERIVED_BUFFER_SUFFIXES)
        ]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"{path} does not match {self.spec.model_id}: missing "
                f"{missing[:5]} ({len(missing)} total), unexpected "
                f"{list(result.unexpected_keys)[:5]} ({len(result.unexpected_keys)} total)"
            )

    def load_model(self, model_name: str) -> None:
        """Build the tokenizer and model and load the verified checkpoint."""
        path = self.verify_weights()
        self.tokenizer = self.build_tokenizer()
        model: Any = GPTNeoXForCausalLM(self.build_config())  # type: ignore[no-untyped-call]  # transformers model __init__ is untyped
        self.load_state_dict_strict(model, path)
        model.eval()
        self.model = model.to(self.device)

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Codon-tokenize, truncate to the context, and pad to the longest sequence."""
        encoding = self.tokenizer(
            [codon_tokens(s) for s in sequences],
            max_length=self.max_sequence_size,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    @torch.no_grad()
    def embed(
        self, sequences: str | list[str], mean_embedding: bool = False
    ) -> torch.Tensor:
        """Last-layer hidden states ``[batch, codons, hidden]``, or their masked mean
        ``[batch, hidden]`` when ``mean_embedding`` is set.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        batch = self.tokenize(sequences)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs: Any = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden: torch.Tensor = outputs.hidden_states[-1].detach()
        if not mean_embedding:
            return hidden
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    model = GenSLM("genslm_25M_patric")
    print(json.dumps(model.spec.model_dump(), indent=2))
    sample = "ATGGCCGTCAAGAAACGGGGTTAA"
    print(model.embed(sample, mean_embedding=True).shape)
