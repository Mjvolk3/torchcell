# tests/torchcell/benchmark/_fake_embedder.py
# [[tests.torchcell.benchmark._fake_embedder]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/_fake_embedder.py
"""A deterministic stand-in for the Model2Vec model of the hybrid search.

Each axis of the vector is a concept with a few keywords; a text's value on an axis is
1 when the lowercased text contains any of the keywords and 0 otherwise. So a card
about cell shape is ``[1, 0, 0]``, the query ``"epistasis"`` is ``[0, 1, 0]`` and a
text with none of the keywords is the zero vector, whose cosine with anything is 0.
Every call is logged, so a test can check that cards are embedded once.
"""

import numpy as np
import numpy.typing as npt

AXES: tuple[tuple[str, ...], ...] = (
    ("shape", "morphology"),
    ("interaction", "epistasis"),
    ("essential", "inviable"),
)


class FakeEmbedder:
    """Concept-keyword vectors, one axis per entry of :data:`AXES`."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> npt.NDArray[np.float64]:
        """One row per text: 1.0 on each axis whose keywords the text contains."""
        self.calls.append(list(texts))
        return np.array(
            [
                [
                    float(any(word in text.lower() for word in keywords))
                    for keywords in AXES
                ]
                for text in texts
            ],
            dtype=np.float64,
        )
