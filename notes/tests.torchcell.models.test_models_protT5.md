---
id: wj258ktd7tdz0jx9kj7bnd0
title: test_models_protT5
desc: ''
updated: 1791269247536
created: 1791269247536
---

## 2026.10.06 - Phase 21 tests

- Fixture: `T5Tokenizer` / `T5EncoderModel` replaced by recorders; fake encoder records `to`, `half`, `full`.
- `_prepare_sequence`: U, Z, O, B to X; symbols space-separated.
- Finding (protT5.py:75): `torch.device("cpu") == "cpu"` is False, so the CPU model is cast to fp16, opposite of the comment; the other branch calls `full()`, which `nn.Module` lacks.
- Finding (protT5.py:99): unmasked `mean(dim=1)`; `"AC"` pools to 1.0 alone and 2.0 when batched with `"ACGT"`. `ProtT5Dataset` embeds one sequence per call, so stored vectors include the EOS position but no padding.
- `max_sequence_size` (40000) is never applied by `embed`.
- Reach (audit 1): both findings are latent; no stored number is known to depend on them.
- Audit 1 revision: line cites corrected to protT5.py:75 and :99; the fake encoder's output now carries a grad-requiring weight, and both the per-token and mean outputs are asserted `requires_grad is False` (pins the `torch.no_grad` at protT5.py:90).
