---
id: cb0f1us2g1f4dr9s7c58751
title: Test_llm
desc: ''
updated: 1790777416393
created: 1790777416393
---

## 2026.09.30 - Phase 17: two abstract bases and a container

New file, five functions (11 cases), 63 to 100 percent. The module is only two abstract base classes and an attrs container: `__init__` resets tokenizer and model then calls `load_model` once; `max_sequence_size` returns 0 as 0 and refuses None with the exact message; the exact abstract-class `TypeError` text in four variants; `pretrained_LLM` field order and equality. Finding: the abstract bodies are `pass`, so `super().embed()` returns None instead of raising (lines 42-92).

## 2026.10.01 - Abstract bodies raise (issue #541)

The None-returning Finding is retired: each of the three abstract bodies on both bases raises `NotImplementedError` with the exact message.
