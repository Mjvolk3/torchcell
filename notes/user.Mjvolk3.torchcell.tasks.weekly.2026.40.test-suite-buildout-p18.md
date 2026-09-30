---
id: 2jz7pykm13ywm3jpzcnop0g
title: test-suite-buildout-p18
desc: ''
updated: 1790780627184
created: 1790780627184
---
- [x] PR-18 of [[plan.test-suite-buildout.2026.09.25]]: the six embedding datasets on a faked backbone ([[tests.torchcell.datasets.test_esm2]], [[tests.torchcell.datasets.test_protT5]], [[tests.torchcell.datasets.test_datasets_nucleotide_transformer]], [[tests.torchcell.datasets.test_datasets_fungal_up_down_transformer]], [[tests.torchcell.datasets.test_codon_language_model]], [[tests.torchcell.datasets.test_random_embedding]]), the two model wrappers and the two GNN layers ([[tests.torchcell.models.test_nucleotide_transformer]], [[tests.torchcell.models.test_fungal_up_down_transformer]], [[tests.torchcell.models.test_graph_convolution]], [[tests.torchcell.models.test_graph_attention]]), the Ohya adapter and the OCR helpers ([[tests.torchcell.adapters.test_ohya2005_adapter]], [[tests.torchcell.literature.test_ocr]]); three Opus writers, one Fable audit (0 rejected, 1 rewritten); 22 findings pinned, a fresh Nucleotide Transformer build raises and both GNN classes cannot be constructed as shipped ([[test-campaign.2026.09.25]]).
